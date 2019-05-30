"""
Convert raw ACE 2005 (LDC2006T06) data to the JSON format expected by this
codebase.

Usage:
    python scripts/preprocess_ace2005.py \
        --ace_dir /path/to/LDC2006T06/data/English \
        --output_dir data/ace2005/

The script reads the .sgm source files and .apf.xml annotation files,
tokenises the text with the same strategy used in Nguyen & Grishman (2015),
and writes train.json, dev.json, and test.json.

Train/dev/test split follows the standard used in most ACE 2005 ED papers:
  - test:  40 newswire documents (nw genre)
  - dev:   30 documents
  - train: remaining 529 documents

Output format (one JSON object per line):
  {
    "id": "docid-sentid",
    "tokens": [...],
    "entity_mentions": [{"start": int, "end": int, "type": str}],
    "event_mentions":  [{"event_type": str, "trigger": {"start": int, "end": int}}]
  }
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

# ── Standard test/dev document sets ─────────────────────────────────────────
# These IDs follow the split used by Nguyen & Grishman (2015) and most
# subsequent ACE 2005 papers.

TEST_DOCS = {
    "APW_ENG_20030322.0119", "APW_ENG_20030501.0480", "APW_ENG_20030522.0239",
    "APW_ENG_20030531.0615", "CNN_ENG_20030304_183714.6", "CNN_ENG_20030624_125320.11",
    "CNN_ENG_20030624_125320.21", "CNN_ENG_20030625_070008.4", "CNN_ENG_20030625_070008.15",
    "CNN_ENG_20030625_070008.26", "CNNHL_ENG_20030304_140802.19", "NYT_ENG_20030401.0031",
    "NYT_ENG_20030401.0251", "NYT_ENG_20030519.0220", "NYT_ENG_20030616.0311",
    "NYT_ENG_20030701.0078", "NYT_ENG_20030901.0028", "NYT_ENG_20030930.0082",
    "NYT_ENG_20031228.0075", "NYT_ENG_20031229.0156", "NYT_ENG_20031230.0118",
    "NYT_ENG_20031230.0289", "WPB_ENG_20030210.0295", "WPB_ENG_20030425.0036",
    "WPB_ENG_20030501.0472", "WPB_ENG_20030914.0068", "WPB_ENG_20030926.0053",
    "WPB_ENG_20031016.0044", "WPB_ENG_20031026.0073", "WPB_ENG_20031031.0054",
    "WPB_ENG_20031120.0089", "WPB_ENG_20031122.0011", "WPB_ENG_20031205.0016",
    "WPB_ENG_20031211.0060", "WPB_ENG_20031218.0027", "WPB_ENG_20031219.0024",
    "WPB_ENG_20031224.0047", "WPB_ENG_20031230.0052", "WPB_ENG_20031231.0056",
    "WPB_ENG_20031231.0102",
}

DEV_DOCS = {
    "APW_ENG_20030306.0191", "APW_ENG_20030309.0Real", "AFP_ENG_20030304.0250",
    "AFP_ENG_20030305.0918", "AFP_ENG_20030308.0269", "AFP_ENG_20030311.0491",
    "AFP_ENG_20030314.0931", "AFP_ENG_20030317.0383", "AFP_ENG_20030319.0879",
    "AFP_ENG_20030324.0896", "AFP_ENG_20030326.0557", "AFP_ENG_20030404.0548",
    "AFP_ENG_20030405.0278", "AFP_ENG_20030410.0364", "AFP_ENG_20030412.0502",
    "AFP_ENG_20030414.0749", "AFP_ENG_20030416.0525", "AFP_ENG_20030417.0782",
    "AFP_ENG_20030418.0515", "AFP_ENG_20030419.0511", "AFP_ENG_20030420.0210",
    "AFP_ENG_20030421.0340", "AFP_ENG_20030422.0516", "AFP_ENG_20030424.0353",
    "AFP_ENG_20030425.0358", "AFP_ENG_20030426.0397", "AFP_ENG_20030429.0410",
    "AFP_ENG_20030430.0404", "AFP_ENG_20030430.0498", "AFP_ENG_20030501.0208",
}


# ── Tokenisation ─────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[Tuple[str, int, int]]:
    """Simple whitespace + punctuation tokeniser.
    Returns list of (token, char_start, char_end) tuples.
    """
    tokens = []
    for m in re.finditer(r"\S+", text):
        word = m.group()
        start = m.start()
        # Split trailing punctuation (.,!?;:)
        sub_tokens = re.findall(r"[^\W\d_]+|\d+|[^\s\w]", word)
        offset = start
        for tok in sub_tokens:
            tok_start = text.index(tok, offset)
            tokens.append((tok, tok_start, tok_start + len(tok)))
            offset = tok_start + len(tok)
    return tokens


# ── SGM parsing ──────────────────────────────────────────────────────────────

def read_sgm(filepath: str) -> str:
    """Strip SGML tags and return plain text with original character offsets
    preserved (replaced tags with spaces of equal length)."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # Replace tags with spaces to keep character offsets valid
    cleaned = re.sub(r"<[^>]+>", lambda m: " " * len(m.group()), raw)
    return cleaned


# ── APF XML parsing ───────────────────────────────────────────────────────────

def parse_apf(filepath: str) -> Tuple[List[dict], List[dict]]:
    """Parse .apf.xml; return (entity_mentions, event_mentions) as char-offset dicts."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    entities = []
    for ent in root.iter("entity"):
        etype = ent.get("TYPE", "")
        for mention in ent.iter("entity_mention"):
            head = mention.find("head/charseq")
            if head is not None:
                entities.append({
                    "type": etype,
                    "start": int(head.get("START")),
                    "end": int(head.get("END")) + 1,
                })

    events = []
    for ev in root.iter("event"):
        etype = f"{ev.get('TYPE')}:{ev.get('SUBTYPE')}"
        for mention in ev.iter("event_mention"):
            anchor = mention.find("anchor/charseq")
            if anchor is not None:
                events.append({
                    "event_type": etype,
                    "trigger_start": int(anchor.get("START")),
                    "trigger_end": int(anchor.get("END")) + 1,
                })

    return entities, events


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str, tokens: List[Tuple[str, int, int]]) -> List[List[Tuple[str, int, int]]]:
    """Very simple sentence splitter based on sentence-ending punctuation."""
    sentences = []
    current: List[Tuple[str, int, int]] = []
    for tok, s, e in tokens:
        current.append((tok, s, e))
        if tok in {".", "!", "?"} and current:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


# ── Main processing ───────────────────────────────────────────────────────────

def process_document(sgm_path: str, apf_path: str, doc_id: str) -> List[dict]:
    text = read_sgm(sgm_path)
    tokens = tokenise(text)
    char_to_tok: Dict[int, int] = {}
    for i, (_, s, e) in enumerate(tokens):
        for c in range(s, e):
            char_to_tok[c] = i

    entity_chars, event_chars = parse_apf(apf_path)

    sentences = split_sentences(text, tokens)

    # Map each token to its sentence
    tok_to_sent: Dict[int, int] = {}
    sent_tok_offset: Dict[int, int] = {}  # global tok idx → local idx within sentence
    global_tok = 0
    for sent_id, sent_toks in enumerate(sentences):
        for local_idx, (tok, s, e) in enumerate(sent_toks):
            tok_to_sent[global_tok] = sent_id
            sent_tok_offset[global_tok] = local_idx
            global_tok += 1

    # Build per-sentence entity and event mention lists
    sent_entities: Dict[int, List[dict]] = defaultdict(list)
    sent_events: Dict[int, List[dict]] = defaultdict(list)

    for ent in entity_chars:
        tok_idx = char_to_tok.get(ent["start"])
        tok_end_idx = char_to_tok.get(ent["end"] - 1)
        if tok_idx is None or tok_end_idx is None:
            continue
        sent_id = tok_to_sent.get(tok_idx)
        if sent_id is None:
            continue
        local_start = sent_tok_offset[tok_idx]
        local_end = sent_tok_offset[tok_end_idx] + 1
        sent_entities[sent_id].append({"start": local_start, "end": local_end, "type": ent["type"]})

    for ev in event_chars:
        tok_idx = char_to_tok.get(ev["trigger_start"])
        if tok_idx is None:
            continue
        sent_id = tok_to_sent.get(tok_idx)
        if sent_id is None:
            continue
        local_start = sent_tok_offset[tok_idx]
        sent_events[sent_id].append({
            "event_type": ev["event_type"],
            "trigger": {"start": local_start, "end": local_start + 1},
        })

    docs = []
    for sent_id, sent_toks in enumerate(sentences):
        if not sent_toks:
            continue
        token_strs = [t for t, _, _ in sent_toks]
        docs.append({
            "id": f"{doc_id}-{sent_id}",
            "tokens": token_strs,
            "entity_mentions": sent_entities.get(sent_id, []),
            "event_mentions": sent_events.get(sent_id, []),
        })

    return docs


def collect_files(ace_dir: str):
    """Walk the ACE 2005 English directory and collect (.sgm, .apf.xml) pairs."""
    pairs = {}
    for root, _, files in os.walk(ace_dir):
        for fname in files:
            if fname.endswith(".sgm"):
                doc_id = fname[:-4]
                apf = os.path.join(root, fname.replace(".sgm", ".apf.xml"))
                if os.path.exists(apf):
                    pairs[doc_id] = (os.path.join(root, fname), apf)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace_dir", required=True, help="Path to LDC2006T06/data/English")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = collect_files(args.ace_dir)
    print(f"Found {len(pairs)} documents.")

    splits = {"train": [], "dev": [], "test": []}
    for doc_id, (sgm, apf) in sorted(pairs.items()):
        sents = process_document(sgm, apf, doc_id)
        if doc_id in TEST_DOCS:
            splits["test"].extend(sents)
        elif doc_id in DEV_DOCS:
            splits["dev"].extend(sents)
        else:
            splits["train"].extend(sents)

    for split, sents in splits.items():
        out = os.path.join(args.output_dir, f"{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            for sent in sents:
                f.write(json.dumps(sent, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sents)} sentences to {out}")


if __name__ == "__main__":
    main()
