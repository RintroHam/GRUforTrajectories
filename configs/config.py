# configs/config.py
# 配置参数
from dataclasses import dataclass

@dataclass
class ModelConfig:
    timestep: int = 15
    batch_size: int = 10
    featrue_size: int = 2
    hidden_size: int = 256
    output_size: int = 2
    num_layers: int = 1
    epochs: int = 50
    learning_rate: float = 0.01
    train_population: float = 0.8