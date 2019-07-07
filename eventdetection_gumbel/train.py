"""Training loop for LearnToSelectED."""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from tqdm import tqdm

from .config import Config
from .data import (
    ACE2005_NER_TAGS,
    Vocabulary,
    build_dataset,
    get_label_space,
    make_dataloader,
)
from .evaluate import format_results, run_evaluation
from .model import LearnToSelectED

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Config, vocab: Vocabulary, n_classes: int) -> LearnToSelectED:
    pretrained = (
        torch.tensor(vocab.embeddings, dtype=torch.float)
        if vocab.embeddings is not None
        else None
    )
    mcfg = config.model
    model = LearnToSelectED(
        vocab_size=vocab.size,
        n_classes=n_classes,
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
        pretrained_embeddings=pretrained,
    )
    if config.train.freeze_embeddings:
        model.freeze_embeddings()
    return model


def train(config: Config):
    set_seed(config.train.seed)

    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    logger.info(f"Using device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Label space ──────────────────────────────────────────────────────────
    event_types = get_label_space(config.data.dataset)
    label2id = {et: i for i, et in enumerate(event_types)}
    n_classes = len(event_types)
    logger.info(f"Label space: {n_classes} classes ({config.data.dataset})")

    # ── NER tag space ────────────────────────────────────────────────────────
    ner_tag2id = {tag: i for i, tag in enumerate(ACE2005_NER_TAGS)}

    # ── Vocabulary & embeddings ──────────────────────────────────────────────
    vocab = Vocabulary.from_word2vec(
        config.data.word2vec_file, embed_dim=config.model.embed_dim
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = build_dataset(
        config.data.train_file, vocab, label2id, ner_tag2id,
        max_position=config.model.max_position,
    )
    dev_ds = build_dataset(
        config.data.dev_file, vocab, label2id, ner_tag2id,
        max_position=config.model.max_position,
    )

    train_loader = make_dataloader(train_ds, config.train.batch_size, shuffle=True)
    dev_loader = make_dataloader(dev_ds, config.train.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config, vocab, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.train.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1 = 0.0
    patience_count = 0
    best_checkpoint = os.path.join(config.output_dir, "best_model.pt")

    for epoch in range(1, config.train.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            word_ids = batch["word_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            ner_ids = batch["ner_ids"].to(device)
            anchor_idx = batch["anchor_idx"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(word_ids, pos_ids, ner_ids, anchor_idx, lengths, mask)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        precision, recall, f1, acc = run_evaluation(model, dev_loader, device)
        logger.info(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | dev {format_results(precision, recall, f1, acc)}"
        )

        if f1 > best_f1:
            best_f1 = f1
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "f1": f1,
                    "config": config,
                    "label2id": label2id,
                    "ner_tag2id": ner_tag2id,
                    "vocab_size": vocab.size,
                },
                best_checkpoint,
            )
            logger.info(f"  ✓ New best F1: {f1 * 100:.1f} — saved to {best_checkpoint}")
        else:
            patience_count += 1
            if patience_count >= config.train.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience_count} epochs).")
                break

    logger.info(f"Training complete. Best dev F1: {best_f1 * 100:.1f}")
    return best_checkpoint


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Train LearnToSelectED")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    config = Config.from_dict(cfg_dict)
    train(config)


if __name__ == "__main__":
    main()
