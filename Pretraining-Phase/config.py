import yaml
from pydantic import BaseModel
from typing import Optional


class DataConfig(BaseModel):
    path: str
    context_length: int
    batch_size: int


class TrainingConfig(BaseModel):
    precision: str
    max_iters: int
    eval_iters: int
    log_interval: int
    device: str
    grad_clip: float
    learning_rate: float
    warmup_iters: int
    min_lr: float
    weight_decay: float


class ModelConfig(BaseModel):
    n_block: int
    n_head: int
    d_model: int
    dropout: float
    best_model: str


class SaveConfig(BaseModel):
    ckpt_dir: str


class GenerationConfig(BaseModel):
    max_new_token: int
    top_k: int
    temperature: float


class GPTConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    save: SaveConfig
    generation: GenerationConfig


def load_config(path: str = "config.yaml") -> GPTConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        print("Loading the config file...")
    return GPTConfig(**cfg_dict)