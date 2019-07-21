"""Evaluate a saved checkpoint on a test split."""

import argparse
import logging

import torch
import yaml

from eventdetection_gumbel.config import Config
from eventdetection_gumbel.data import (
    Vocabulary,
    build_dataset,
    make_dataloader,
    get_label_space,
    ACE2005_NER_TAGS,
)
from eventdetection_gumbel.evaluate import format_results, run_evaluation
from eventdetection_gumbel.model import LearnToSelectED

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LearnToSelectED")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint (.pt)")
    parser.add_argument("--test_file", required=True, help="Path to test JSON file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config.from_dict(cfg_dict)

    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    label2id = ckpt["label2id"]
    ner_tag2id = ckpt.get("ner_tag2id", {tag: i for i, tag in enumerate(ACE2005_NER_TAGS)})
    vocab_size = ckpt["vocab_size"]

    # Rebuild vocabulary (embeddings not needed for inference)
    vocab = Vocabulary()
    vocab.word2id = {}  # will be populated but embeddings not loaded
    logger.info(f"Loading checkpoint from {args.checkpoint}")

    # Rebuild model
    mcfg = config.model
    model = LearnToSelectED(
        vocab_size=vocab_size,
        n_classes=len(label2id),
        embed_dim=mcfg.embed_dim,
        pos_embed_dim=mcfg.pos_embed_dim,
        ner_embed_dim=mcfg.ner_embed_dim,
        lstm_hidden=mcfg.lstm_hidden,
        compose_dim=mcfg.compose_dim,
        n_steps=mcfg.n_steps,
        temperature=mcfg.temperature,
        dropout=mcfg.dropout,
        max_position=mcfg.max_position,
        n_ner_tags=mcfg.n_ner_tags,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Model loaded.")

    # Build test dataset
    # For evaluation we need the vocabulary mapping from training, so we
    # load embeddings the same way as training (or accept degraded vocab).
    vocab_full = Vocabulary.from_word2vec(
        config.data.word2vec_file, embed_dim=mcfg.embed_dim
    )

    test_ds = build_dataset(
        args.test_file, vocab_full, label2id, ner_tag2id,
        max_position=mcfg.max_position,
    )
    test_loader = make_dataloader(test_ds, config.train.batch_size, shuffle=False)

    precision, recall, f1, acc = run_evaluation(model, test_loader, device)
    print(f"\nTest Results — {format_results(precision, recall, f1, acc)}")


if __name__ == "__main__":
    main()
