"""
Data loading and preprocessing for event detection.

Expected input format: JSON lines, one document per line.
Each document:
  {
    "id": str,
    "tokens": List[str],
    "entity_mentions": [{"start": int, "end": int, "type": str}],
    "event_mentions": [{"event_type": str, "trigger": {"start": int, "end": int}}]
  }
Token indices are 0-based; start inclusive, end exclusive.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Label spaces ────────────────────────────────────────────────────────────

ACE2005_EVENT_TYPES = [
    "Other",
    "Life:Be-Born", "Life:Marry", "Life:Divorce", "Life:Injure", "Life:Die",
    "Movement:Transport",
    "Business:Start-Org", "Business:End-Org", "Business:Declare-Bankruptcy", "Business:Merge-Org",
    "Conflict:Attack", "Conflict:Demonstrate",
    "Contact:Meet", "Contact:Phone-Write",
    "Personnel:Elect", "Personnel:End-Position", "Personnel:Start-Position", "Personnel:Nominate",
    "Justice:Arrest-Jail", "Justice:Release-Parole", "Justice:Trial-Hearing",
    "Justice:Charge-Indict", "Justice:Sue", "Justice:Convict", "Justice:Sentence",
    "Justice:Fine", "Justice:Execute", "Justice:Extradite", "Justice:Acquit",
    "Justice:Appeal", "Justice:Pardon",
    "Transaction:Transfer-Money", "Transaction:Transfer-Ownership",
]  # 34 classes (33 event types + Other)

TAC2015_EVENT_TYPES = [
    "Other",
    "Life:Be-Born", "Life:Marry", "Life:Divorce", "Life:Injure", "Life:Die",
    "Movement:Transport-Artifact", "Movement:Transport-Person",
    "Business:Start-Org", "Business:End-Org", "Business:Declare-Bankruptcy", "Business:Merge-Org",
    "Conflict:Attack", "Conflict:Demonstrate",
    "Contact:Meet", "Contact:Phone-Write", "Contact:Broadcast", "Contact:Contact",
    "Personnel:Elect", "Personnel:End-Position", "Personnel:Start-Position", "Personnel:Nominate",
    "Justice:Arrest-Jail", "Justice:Release-Parole", "Justice:Trial-Hearing",
    "Justice:Charge-Indict", "Justice:Sue", "Justice:Convict", "Justice:Sentence",
    "Justice:Fine", "Justice:Execute", "Justice:Extradite", "Justice:Acquit",
    "Justice:Appeal", "Justice:Pardon",
    "Transaction:Transfer-Money", "Transaction:Transfer-Ownership",
    "Manufacture:Artifact",
    "Inspection:Artifact",
]  # 39 classes (38 event types + Other)

# BIO entity type tags for ACE 2005 (7 entity types × 2 + O)
ACE2005_NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-GPE", "I-GPE",
    "B-FAC", "I-FAC",
    "B-LOC", "I-LOC",
    "B-VEH", "I-VEH",
    "B-WEA", "I-WEA",
]  # 15 tags


def get_label_space(dataset: str) -> List[str]:
    if dataset == "ace2005":
        return ACE2005_EVENT_TYPES
    elif dataset == "tac2015":
        return TAC2015_EVENT_TYPES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Vocabulary ───────────────────────────────────────────────────────────────

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.word2id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.id2word: List[str] = [self.PAD, self.UNK]
        self.embeddings: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return len(self.word2id)

    def add(self, word: str) -> int:
        if word not in self.word2id:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
        return self.word2id[word]

    def __contains__(self, word: str) -> bool:
        return word in self.word2id

    def encode(self, word: str) -> int:
        return self.word2id.get(word, self.word2id[self.UNK])

    def load_word2vec(self, filepath: str, embed_dim: int = 300) -> np.ndarray:
        """Load pretrained word2vec embeddings (binary format).
        Words already in vocab get their pretrained vector; others get random init."""
        try:
            import gensim
            logger.info(f"Loading word2vec from {filepath} ...")
            w2v = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
        except ImportError:
            raise ImportError("gensim is required to load word2vec embeddings.")

        matrix = np.random.uniform(-0.25, 0.25, (self.size, embed_dim)).astype(np.float32)
        matrix[0] = 0.0  # PAD → zero vector

        found = 0
        for word, idx in self.word2id.items():
            if word in w2v:
                matrix[idx] = w2v[word]
                found += 1

        logger.info(f"Loaded pretrained vectors for {found}/{self.size} words.")
        self.embeddings = matrix
        return matrix

    @classmethod
    def from_word2vec(cls, filepath: str, embed_dim: int = 300) -> "Vocabulary":
        """Build vocabulary directly from word2vec keys."""
        try:
            import gensim
            logger.info(f"Building vocabulary from {filepath} ...")
            w2v = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
        except ImportError:
            raise ImportError("gensim is required to load word2vec embeddings.")

        vocab = cls()
        for word in w2v.key_to_index:
            vocab.add(word)

        matrix = np.zeros((vocab.size, embed_dim), dtype=np.float32)
        matrix[1] = np.random.uniform(-0.25, 0.25, embed_dim)  # UNK
        for word, idx in vocab.word2id.items():
            if word in w2v:
                matrix[idx] = w2v[word]

        vocab.embeddings = matrix
        logger.info(f"Vocabulary size: {vocab.size}")
        return vocab


# ── BIO entity tagging ───────────────────────────────────────────────────────

def bio_tag_tokens(tokens: List[str], entity_mentions: List[dict]) -> List[str]:
    """Produce BIO entity type tags for a token sequence."""
    tags = ["O"] * len(tokens)
    for ent in entity_mentions:
        etype = ent["type"]
        start, end = ent["start"], ent["end"]
        for i in range(start, end):
            prefix = "B" if i == start else "I"
            tags[i] = f"{prefix}-{etype}"
    return tags


# ── Example dataclass ────────────────────────────────────────────────────────

@dataclass
class EventExample:
    tokens: List[str]
    ner_tags: List[str]
    anchor_idx: int
    label: int
    doc_id: str = ""


# ── Document → examples ──────────────────────────────────────────────────────

def doc_to_examples(
    doc: dict,
    label2id: Dict[str, int],
    negative_sampling_rate: float = 1.0,
) -> List[EventExample]:
    """Convert a document dict to a list of EventExample.

    Generates one example per event trigger (positive) and optionally
    samples non-trigger words as negative (Other) examples.
    """
    tokens = doc["tokens"]
    entity_mentions = doc.get("entity_mentions", [])
    event_mentions = doc.get("event_mentions", [])
    doc_id = doc.get("id", "")

    ner_tags = bio_tag_tokens(tokens, entity_mentions)

    # Map trigger positions to event types
    trigger_map: Dict[int, str] = {}
    for ev in event_mentions:
        trigger_start = ev["trigger"]["start"]
        trigger_map[trigger_start] = ev["event_type"]

    examples = []

    # Positive examples (trigger words)
    for anchor_idx, event_type in trigger_map.items():
        label = label2id.get(event_type, 0)
        examples.append(EventExample(
            tokens=tokens,
            ner_tags=ner_tags,
            anchor_idx=anchor_idx,
            label=label,
            doc_id=doc_id,
        ))

    # Negative examples (non-trigger words labeled as Other)
    other_id = label2id.get("Other", 0)
    rng = np.random.default_rng()
    for i, token in enumerate(tokens):
        if i in trigger_map:
            continue
        if negative_sampling_rate < 1.0 and rng.random() > negative_sampling_rate:
            continue
        examples.append(EventExample(
            tokens=tokens,
            ner_tags=ner_tags,
            anchor_idx=i,
            label=other_id,
            doc_id=doc_id,
        ))

    return examples


# ── Dataset ──────────────────────────────────────────────────────────────────

class EventDataset(Dataset):
    def __init__(
        self,
        examples: List[EventExample],
        vocab: Vocabulary,
        ner_tag2id: Dict[str, int],
        max_position: int = 200,
    ):
        self.examples = examples
        self.vocab = vocab
        self.ner_tag2id = ner_tag2id
        self.max_position = max_position

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        tokens = ex.tokens
        anchor = ex.anchor_idx
        L = len(tokens)

        word_ids = [self.vocab.encode(t) for t in tokens]

        # Relative position: clamp to [-max_position, max_position]
        pos_ids = [
            max(-self.max_position, min(self.max_position, i - anchor))
            for i in range(L)
        ]

        ner_ids = [
            self.ner_tag2id.get(tag, 0) for tag in ex.ner_tags
        ]

        return {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "ner_ids": torch.tensor(ner_ids, dtype=torch.long),
            "anchor_idx": torch.tensor(anchor, dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.long),
            "length": torch.tensor(L, dtype=torch.long),
        }


def collate_fn(batch: List[dict]) -> dict:
    """Pad sequences to the max length in the batch."""
    max_len = max(b["length"].item() for b in batch)

    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    pos_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    ner_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    anchor_idx = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["length"].item()
        word_ids[i, :L] = b["word_ids"]
        pos_ids[i, :L] = b["pos_ids"]
        ner_ids[i, :L] = b["ner_ids"]
        mask[i, :L] = True
        anchor_idx[i] = b["anchor_idx"]
        labels[i] = b["label"]
        lengths[i] = b["length"]

    return {
        "word_ids": word_ids,
        "pos_ids": pos_ids,
        "ner_ids": ner_ids,
        "mask": mask,
        "anchor_idx": anchor_idx,
        "labels": labels,
        "lengths": lengths,
    }


# ── File loading ─────────────────────────────────────────────────────────────

def load_docs(filepath: str) -> List[dict]:
    docs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs)} documents from {filepath}")
    return docs


def build_dataset(
    filepath: str,
    vocab: Vocabulary,
    label2id: Dict[str, int],
    ner_tag2id: Dict[str, int],
    max_position: int = 200,
    negative_sampling_rate: float = 1.0,
) -> EventDataset:
    docs = load_docs(filepath)
    examples: List[EventExample] = []
    for doc in docs:
        examples.extend(doc_to_examples(doc, label2id, negative_sampling_rate))
    logger.info(f"Created {len(examples)} examples from {filepath}")
    return EventDataset(examples, vocab, ner_tag2id, max_position)


def make_dataloader(
    dataset: EventDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
