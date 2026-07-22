from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "sample"


@dataclass(frozen=True)
class ModelConfig:
    model_version: str = "v1-0-0"
    embedding_dim: int = 128
    latent_dim: int = 64


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 16
    learning_rate: float = 1.893292917167e-4
    epochs: int = 55
    verbose: bool = True
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class InferenceConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: str | None = None
