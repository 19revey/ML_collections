from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, root_validator

__all__ = ["ModelConfigure", "TrainConfigure"]


class ModelConfigure(BaseModel):
    vocab_size: int
    context_length: int = 64
    embedding_size: int = 32
    num_heads: int = 4
    num_blocks: int = 4
    attn_dropout_prob: float = 0.1
    embed_dropout_prob: float = 0.1
    device: str = "auto"



class TrainConfigure(ModelConfigure):
    batch_size: int = 32
    learning_rate: float = 5e-4
    batches_per_eval: int = 100
    eval_interval: int = 500
    num_epochs: int = 3
    checkpoint_path: Union[str, Path] = Path("checkpoints")
    save_all_checkpoints: bool = False
    overwrite_checkpoints: bool = True
