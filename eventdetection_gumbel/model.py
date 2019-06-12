"""
LearnToSelectED: Learning to Select Important Context Words for Event Detection.

Reference: "Learning to Select Important Context Words for Event Detection" (ACL 2019).

Model components:
  1. SentenceEncoder  — BiLSTM over word + position + entity-type embeddings
  2. LSTMCompose      — LSTM-style composition function cm(u, v, c)
  3. GumbelWordSelector — iterative word selector with Straight-Through Gumbel-Softmax
  4. LearnToSelectED  — full model (encoder → selector → classifier)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sentence Encoder ─────────────────────────────────────────────────────────

class SentenceEncoder(nn.Module):
    """BiLSTM encoder.

    Encodes each word as [emb_i, pos_i, ner_i] and runs a bidirectional LSTM
    to produce context-aware hidden states h_1 ... h_n.

    Args:
        vocab_size:   size of word vocabulary (including PAD/UNK).
        embed_dim:    word embedding dimension (300 for word2vec).
        pos_embed_dim: relative-position embedding dimension (50).
        ner_embed_dim: entity-type BIO tag embedding dimension (50).
        hidden_dim:   per-direction LSTM hidden units (600).
        max_position: maximum relative position distance (±200).
        n_ner_tags:   number of BIO entity-type tags (15 for ACE 2005).
        dropout:      applied to input embeddings and LSTM output.
        pretrained_embeddings: optional (vocab_size, embed_dim) tensor.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        pos_embed_dim: int = 50,
        ner_embed_dim: int = 50,
        hidden_dim: int = 600,
        max_position: int = 200,
        n_ner_tags: int = 15,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.max_position = max_position
        self.output_dim = 2 * hidden_dim

        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_emb.weight.data.copy_(pretrained_embeddings)

        # Position embedding table covers offsets in [-max_position, +max_position]
        self.pos_emb = nn.Embedding(2 * max_position + 1, pos_embed_dim)

        self.ner_emb = nn.Embedding(n_ner_tags, ner_embed_dim, padding_idx=0)

        input_dim = embed_dim + pos_embed_dim + ner_embed_dim
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        word_ids: torch.Tensor,   # (B, L)
        pos_ids: torch.Tensor,    # (B, L) — relative positions (i − anchor)
        ner_ids: torch.Tensor,    # (B, L) — BIO entity-type tag ids
        lengths: torch.Tensor,    # (B,)   — true sequence lengths
    ) -> torch.Tensor:            # (B, L, 2*hidden_dim)
        emb = self.word_emb(word_ids)                           # (B, L, 300)
        pos = self.pos_emb(pos_ids + self.max_position)         # (B, L,  50)
        ner = self.ner_emb(ner_ids)                             # (B, L,  50)

        x = self.dropout(torch.cat([emb, pos, ner], dim=-1))   # (B, L, 400)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.bilstm(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.dropout(hidden)   # (B, L, 1200)


# ── LSTM Composition Function ────────────────────────────────────────────────

class LSTMCompose(nn.Module):
    """LSTM-style composition function cm(u, v, c) from Eq. (2) of the paper.

    Given word representation u, current overall representation v, and cell
    state c, computes:
        [f, i, o, g] = [σ, σ, σ, tanh](W_com [u; v] + b_com)
        new_c = f ⊙ c + i ⊙ g
        new_h = o ⊙ tanh(new_c)

    Operates on arbitrary leading dimensions (e.g., (B, D) or (B, L, D)).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(2 * dim, 4 * dim)
        self.dim = dim

    def forward(
        self,
        u: torch.Tensor,   # (..., dim)
        v: torch.Tensor,   # (..., dim)
        c: torch.Tensor,   # (..., dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.W(torch.cat([u, v], dim=-1))         # (..., 4*dim)
        f, i, o, g = gates.chunk(4, dim=-1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


# ── Gumbel Word Selector ─────────────────────────────────────────────────────

class GumbelWordSelector(nn.Module):
    """Iterative word selector using Straight-Through Gumbel-Softmax (STGS).

    Performs k sequential word-selection steps (Section 2.2 of the paper).
    At each step:
      - Computes relevance scores r(w_j) = q^T · cm(u_j, v, c) for all
        non-selected words w_j.
      - Training: samples a word via STGS; uses the soft distribution for
        gradient computation (straight-through estimator).
      - Inference: greedy argmax.

    Args:
        encoder_dim:  output dimension of the BiLSTM encoder (2 * lstm_hidden).
        compose_dim:  internal dimension for composition (300).
        n_steps:      number of word-selection steps k (4).
        temperature:  Gumbel-Softmax temperature τ (1.0).
    """

    def __init__(
        self,
        encoder_dim: int,
        compose_dim: int = 300,
        n_steps: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.temperature = temperature
        self.compose_dim = compose_dim

        # Project BiLSTM hidden states to compose_dim
        self.proj = nn.Linear(encoder_dim, compose_dim)

        self.compose = LSTMCompose(compose_dim)

        # Learnable query vector q
        self.query = nn.Parameter(torch.Tensor(compose_dim))
        nn.init.uniform_(self.query, -0.1, 0.1)

    def _sample_gumbel(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        U = torch.zeros(shape, device=device).uniform_().clamp_(1e-20, 1.0)
        return -torch.log(-torch.log(U))

    def forward(
        self,
        hidden: torch.Tensor,      # (B, L, encoder_dim)
        anchor_idx: torch.Tensor,  # (B,)
        mask: torch.Tensor,        # (B, L) bool — True for valid (non-padding) positions
    ) -> torch.Tensor:             # (B, compose_dim)
        B, L, _ = hidden.shape

        h = self.proj(hidden)   # (B, L, compose_dim)

        # v^1 = h_a  (anchor word hidden state after projection)
        v = h[torch.arange(B, device=hidden.device), anchor_idx]  # (B, D)
        c = torch.zeros_like(v)

        # Track selected positions; anchor is pre-selected
        selected = torch.zeros(B, L, dtype=torch.bool, device=hidden.device)
        selected[torch.arange(B, device=hidden.device), anchor_idx] = True

        for _ in range(self.n_steps):
            available = mask & ~selected   # (B, L)

            # Guard: if every batch element has no available words, stop
            if not available.any():
                break

            # For numerical stability, ensure each row has at least one
            # available position in the softmax (rows with no available words
            # will be handled by masking v/c updates below)
            has_avail = available.any(dim=-1)   # (B,)
            avail_safe = available.clone()
            avail_safe[~has_avail, 0] = True    # dummy availability — prevents NaN

            # Expand v, c over the sequence dimension
            v_exp = v.unsqueeze(1).expand(-1, L, -1)   # (B, L, D)
            c_exp = c.unsqueeze(1).expand(-1, L, -1)   # (B, L, D)

            # Candidate composed vectors for every position
            cand_h, cand_c = self.compose(h, v_exp, c_exp)   # (B, L, D) each

            # Relevance score: q^T · cm(u_j, v, c)
            scores = (cand_h * self.query).sum(-1)            # (B, L)
            scores = scores.masked_fill(~avail_safe, float("-inf"))

            if self.training:
                gumbel = self._sample_gumbel(scores.shape, scores.device)
                noisy = (scores + gumbel) / self.temperature

                soft_w = F.softmax(noisy, dim=-1)     # (B, L)

                # Hard selection index (argmax of noisy scores)
                hard_idx = noisy.argmax(dim=-1)       # (B,)
                one_hot = torch.zeros_like(soft_w)
                one_hot.scatter_(1, hard_idx.unsqueeze(1), 1.0)

                # Straight-through: forward uses hard, backward uses soft
                st_w = soft_w + (one_hot - soft_w).detach()  # (B, L)

                v_new = (st_w.unsqueeze(-1) * cand_h).sum(1)   # (B, D)
                c_new = (st_w.unsqueeze(-1) * cand_c).sum(1)   # (B, D)

            else:
                hard_idx = scores.argmax(dim=-1)
                v_new = cand_h[torch.arange(B, device=hidden.device), hard_idx]
                c_new = cand_c[torch.arange(B, device=hidden.device), hard_idx]

            # Only update state for batch elements that had available words
            v = torch.where(has_avail.unsqueeze(-1), v_new, v)
            c = torch.where(has_avail.unsqueeze(-1), c_new, c)

            # Mark selected word (use hard_idx; only meaningful when has_avail)
            update_rows = torch.arange(B, device=hidden.device)[has_avail]
            selected[update_rows, hard_idx[has_avail]] = True

        return v   # (B, compose_dim)


# ── Full Model ───────────────────────────────────────────────────────────────

class LearnToSelectED(nn.Module):
    """LearnToSelectED: event detection via learned context word selection.

    Args:
        vocab_size:   word vocabulary size.
        n_classes:    number of event types including Other (34 for ACE 2005).
        embed_dim:    word embedding dim (300).
        pos_embed_dim: position embedding dim (50).
        ner_embed_dim: NER tag embedding dim (50).
        lstm_hidden:  per-direction BiLSTM hidden units (600).
        compose_dim:  composition function hidden dim (300).
        n_steps:      word selection steps k (4).
        temperature:  Gumbel-Softmax temperature τ (1.0).
        dropout:      dropout probability (0.5).
        max_position: max absolute relative position (200).
        n_ner_tags:   number of BIO entity-type tags (15 for ACE 2005).
        pretrained_embeddings: optional pretrained word vectors tensor.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 300,
        pos_embed_dim: int = 50,
        ner_embed_dim: int = 50,
        lstm_hidden: int = 600,
        compose_dim: int = 300,
        n_steps: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.5,
        max_position: int = 200,
        n_ner_tags: int = 15,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.encoder = SentenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            pos_embed_dim=pos_embed_dim,
            ner_embed_dim=ner_embed_dim,
            hidden_dim=lstm_hidden,
            max_position=max_position,
            n_ner_tags=n_ner_tags,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
        )

        self.selector = GumbelWordSelector(
            encoder_dim=2 * lstm_hidden,
            compose_dim=compose_dim,
            n_steps=n_steps,
            temperature=temperature,
        )

        self.classifier = nn.Linear(compose_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        word_ids: torch.Tensor,    # (B, L)
        pos_ids: torch.Tensor,     # (B, L)
        ner_ids: torch.Tensor,     # (B, L)
        anchor_idx: torch.Tensor,  # (B,)
        lengths: torch.Tensor,     # (B,)
        mask: torch.Tensor,        # (B, L) bool
    ) -> torch.Tensor:             # (B, n_classes) logits
        hidden = self.encoder(word_ids, pos_ids, ner_ids, lengths)
        v = self.selector(hidden, anchor_idx, mask)
        logits = self.classifier(self.dropout(v))
        return logits

    def freeze_embeddings(self):
        """Freeze pretrained word embeddings (default behaviour)."""
        self.encoder.word_emb.weight.requires_grad_(False)

    def unfreeze_embeddings(self):
        self.encoder.word_emb.weight.requires_grad_(True)
