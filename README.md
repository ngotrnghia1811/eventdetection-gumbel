# eventdetection-gumbel

Implementation of **"Learning to Select Important Context Words for Event Detection"** (PAKDD 2020).

[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206272/)

## Overview

Event Detection (ED) requires identifying trigger words in text and classifying them into predefined event types. A core challenge is modeling the right context words — different words surrounding the same trigger can change its event type entirely.

This work proposes **LearnToSelectED**: instead of using hand-crafted selection rules (entity mentions, dependency edges), the model *learns* which context words are relevant. It iteratively selects words one at a time, with each selection conditioned on the words already chosen, using the **Straight-Through Gumbel-Softmax** estimator to handle the discrete selection as differentiable during training.

**Architecture:**
1. A bidirectional LSTM encodes the sentence into hidden states using word embeddings (word2vec), relative position embeddings, and entity-type (BIO) embeddings.
2. An iterative selector performs *k = 4* word-selection steps, starting from the trigger candidate's hidden state. At each step, relevance scores are computed via an LSTM-style composition function and the word with the highest score is selected using Gumbel-Softmax.
3. The final accumulated representation is fed into a linear classifier.

The approach implements *hard attention* (only selected words contribute to the representation), as opposed to the soft attention used in most prior work.

## Setup

```bash
conda create -n ed-gumbel python=3.7
conda activate ed-gumbel
pip install -r requirements.txt
pip install -e .
```

## Data

See [data/README.md](data/README.md) for instructions on obtaining and preprocessing ACE 2005 (LDC2006T06) and TAC KBP 2015.

```bash
python scripts/preprocess_ace2005.py \
    --ace_dir /path/to/LDC2006T06/data/English \
    --output_dir data/ace2005/
```

Word2vec embeddings (Google News, 300d): https://code.google.com/archive/p/word2vec/

## Training

```bash
# ACE 2005
bash scripts/train_ace2005.sh

# TAC KBP 2015
bash scripts/train_tac2015.sh
```

Or directly:

```bash
python -m eventdetection_gumbel.train --config configs/ace2005.yaml
```

Key hyperparameters (from `configs/ace2005.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lstm_hidden` | 600 | Per-direction BiLSTM units |
| `compose_dim` | 300 | Composition function hidden dim |
| `n_steps` | 4 | Word selection steps *k* |
| `temperature` | 1.0 | Gumbel-Softmax temperature τ |
| `learning_rate` | 8e-5 | Adam learning rate |
| `batch_size` | 64 | |

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ace2005/best_model.pt \
    --test_file  data/ace2005/test.json \
    --config     configs/ace2005.yaml
```

## Results

Results on the ACE 2005 test set (Table 1 of the paper):

| Model | P | R | F1 |
|-------|:-:|:-:|:--:|
| CNN (Nguyen & Grishman, 2015) | 71.8 | 66.4 | 69.0 |
| JRNN (Nguyen & Grishman, 2016) | 66.0 | 73.0 | 69.3 |
| CNN-LSTM (Feng et al., 2016) | 84.6 | 64.9 | 73.4 |
| GCNN-ENT (Nguyen & Grishman, 2018) | 77.9 | 68.8 | 73.1 |
| SELF-GAN (Hong et al., 2018) | 71.3 | 74.7 | 73.0 |
| **LearnToSelectED (ours)** | **75.4** | **73.1** | **74.2** |

Results on TAC KBP 2015 (Table 2):

| Model | P | R | F1 |
|-------|:-:|:-:|:--:|
| TAC TOP system | 75.2 | 47.7 | 58.4 |
| GCNN-ENT (Nguyen & Grishman, 2018) | 70.3 | 50.6 | 58.8 |
| **LearnToSelectED (ours)** | **63.7** | **58.7** | **61.1** |

Cross-domain F1 on ACE 2005 (Table 3), trained on `bn+nw`:

| Model | in-domain | bc | cts | wl | un |
|-------|:---------:|:--:|:---:|:--:|:--:|
| NCNN | 70.4 | 68.8 | 63.6 | 47.4 | 58.1 |
| SELF-GAN | 69.5 | 68.9 | 63.3 | 50.0 | — |
| **LearnToSelectED** | **70.9** | **69.8** | **64.8** | **50.2** | **59.6** |

## Citation

```bibtex
@inproceedings{ngo-etal-2020-learning,
    title     = "Learning to Select Important Context Words for Event Detection",
    author    = "Ngo, Nghia Trung and Nguyen, Tuan Ngo and Nguyen, Thien Huu",
    booktitle = "Advances in Knowledge Discovery and Data Mining (PAKDD 2020)",
    series    = "Lecture Notes in Computer Science",
    volume    = "12085",
    pages     = "756--768",
    year      = "2020",
    publisher = "Springer, Cham",
    doi       = "10.1007/978-3-030-47436-2_57",
    url       = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7206272/",
}
```

## License

MIT
