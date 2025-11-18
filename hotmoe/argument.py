from dataclasses import dataclass, field
from typing import List, Literal, Optional

from transformers import TrainingArguments


@dataclass
class HotMoEModelArguments:
    model: str = field(default="Qwen/Qwen2-1.5B")
    use_deepspeed: bool = field(default=False)

    # LoRA
    target_modules: Optional[List[str]] = field(default=None)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)

    # MoE routing strategy
    top_k_routing_strategy: bool = field(default=False)
    top_k: int = field(default=2)
    attn_dim: int = field(default=32)
    expert_vector_dim: int = field(default=256)

    # aux loss
    use_div_loss: bool = field(default=False)
    use_sim_loss: bool = field(default=True)
    lambda_auxiliary: float = field(default=0.01)
    beta_s: float = field(default=1.0)
    beta_d: float = field(default=1.5)
    lambda_lm: float = field(default=1.0)
    

    num_experts: int = field(default=8)

    dropout: float = field(default=0.1)


@dataclass
class HotMoEDataArguments:
    dataset_json_path: str = field(
        default="./dataset_paths.json",
        metadata={"help": "JSON file mapping task names to dataset paths"},
    )
    max_seq_length: int = field(default=256)
    preprocessing_num_workers: int = field(default=8)
    data_nums: Optional[int] = field(
        default=1000, metadata={"help": "Data sample number for each task."}
    )


@dataclass
class HotMoETrainingArguments(TrainingArguments):
    seed: int = field(default=42)
    save_dir: str = field(default="./output/")
    num_training_steps: int = field(
        default=1000, metadata={"help": "Total number of training steps."}
    )
    warmup_steps: int = field(
        default=200,
        metadata={"help": "Number of warmup steps for learning rate scheduler."},
    )
    # Scheduler params
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = field(default="constant", metadata={"help": "The scheduler type to use."})
    polynomial_power: int = field(
        default=2, metadata={"help": "Polynomial power for polynomial scheduler."}
    )
    num_cycles: int = field(
        default=5,
        metadata={"help": "Number of cycles for cosine_with_restarts scheduler."},
    )
