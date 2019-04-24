"""Configuration dataclasses for LearnToSelectED."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 50000
    n_classes: int = 34            # 33 ACE event subtypes + Other
    embed_dim: int = 300           # word2vec embedding dimension
    pos_embed_dim: int = 50        # relative-position embedding dimension
    ner_embed_dim: int = 50        # entity-type (BIO) embedding dimension
    lstm_hidden: int = 600         # per-direction BiLSTM hidden units
    compose_dim: int = 300         # composition function hidden dimension
    n_steps: int = 4               # word selection steps k
    temperature: float = 1.0      # Gumbel-Softmax temperature τ
    dropout: float = 0.5
    max_position: int = 200        # maximum relative position distance
    n_ner_tags: int = 15           # BIO tags for ACE 2005 (7 types × 2 + O)


@dataclass
class TrainConfig:
    learning_rate: float = 8e-5
    batch_size: int = 64
    n_epochs: int = 60
    patience: int = 10             # early stopping patience (epochs)
    max_grad_norm: float = 5.0
    seed: int = 42
    freeze_embeddings: bool = True


@dataclass
class DataConfig:
    train_file: str = "data/ace2005/train.json"
    dev_file: str = "data/ace2005/dev.json"
    test_file: str = "data/ace2005/test.json"
    word2vec_file: str = "data/embeddings/GoogleNews-vectors-negative300.bin"
    dataset: str = "ace2005"       # "ace2005" or "tac2015"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints/"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        model_cfg = ModelConfig(**d.get("model", {}))
        train_cfg = TrainConfig(**d.get("train", {}))
        data_cfg = DataConfig(**d.get("data", {}))
        return cls(
            model=model_cfg,
            train=train_cfg,
            data=data_cfg,
            output_dir=d.get("output_dir", "checkpoints/"),
            device=d.get("device", "cuda"),
        )
