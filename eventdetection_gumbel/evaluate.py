"""Evaluation utilities for event detection (P / R / F1)."""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .model import LearnToSelectED


def precision_recall_f1(
    gold: List[int],
    pred: List[int],
    n_classes: int,
    ignore_label: int = 0,
) -> Tuple[float, float, float]:
    """Micro-averaged P/R/F1 over all classes except ignore_label (Other=0).

    Follows the standard ACE 2005 evaluation: a predicted trigger is correct
    if both the offsets and the event subtype match the gold annotation.
    Since this function operates on pre-extracted anchor examples, each
    example is already tied to an anchor position — correctness reduces to
    whether the predicted class matches the gold class (excluding Other).
    """
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        if g != ignore_label and p != ignore_label:
            if g == p:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif g != ignore_label and p == ignore_label:
            fn += 1
        elif g == ignore_label and p != ignore_label:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


@torch.no_grad()
def run_evaluation(
    model: LearnToSelectED,
    dataloader: DataLoader,
    device: torch.device,
    ignore_label: int = 0,
) -> Tuple[float, float, float, float]:
    """Run model inference and compute P/R/F1 + accuracy.

    Returns:
        (precision, recall, f1, accuracy)
    """
    model.eval()

    all_gold: List[int] = []
    all_pred: List[int] = []

    for batch in dataloader:
        word_ids = batch["word_ids"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        ner_ids = batch["ner_ids"].to(device)
        anchor_idx = batch["anchor_idx"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"]

        logits = model(word_ids, pos_ids, ner_ids, anchor_idx, lengths, mask)
        preds = logits.argmax(dim=-1).cpu()

        all_gold.extend(labels.tolist())
        all_pred.extend(preds.tolist())

    n_classes = model.classifier.out_features
    precision, recall, f1 = precision_recall_f1(all_gold, all_pred, n_classes, ignore_label)

    total = len(all_gold)
    correct = sum(g == p for g, p in zip(all_gold, all_pred))
    accuracy = correct / total if total > 0 else 0.0

    return precision, recall, f1, accuracy


def format_results(
    precision: float, recall: float, f1: float, accuracy: float
) -> str:
    return (
        f"P: {precision * 100:.1f}  "
        f"R: {recall * 100:.1f}  "
        f"F1: {f1 * 100:.1f}  "
        f"Acc: {accuracy * 100:.1f}"
    )
