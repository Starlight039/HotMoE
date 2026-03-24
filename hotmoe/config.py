from dataclasses import asdict, dataclass
from typing import Dict, List

import torch

SUPPORTED_CAUSAL_MODELS = "qwen2"

TARGET_MODULE_TYPE = {
    "qwen2": {
        "q": ["q_proj"],
        "k": ["k_proj"],
        "v": ["v_proj"],
        "o": ["o_proj"],
        "wi": ["gate_proj", "up_proj"],
        "wo": ["down_proj"],
        "atte": "self_attn",
        "ffn": "mlp",
        "embed": "embed_tokens",
        "decoders": "model.layers",
    },
}


@dataclass
class HotmoeConfig:
    target_modules: List[str] = None
    peft_type: str = "hotmoe"
    hidden_size: int = None
    model_type: str = None
    torch_dtype: torch.dtype = torch.float32
    eval_mode: bool = False
    dropout: float = 0.1
    max_llm_layer: int = 0

    # routing strategy
    top_k_routing_strategy: bool = False
    top_k: int = 2

    # router
    use_EE_router: bool = False
    use_TE_router: bool = False
    attn_dim: int = 32
    expert_vector_dim: int = 256

    # loss
    use_load_balancing_loss: bool = False
    use_div_loss: bool = False
    use_sim_loss: bool = True
    beta_s: float = 1.0
    beta_d: float = 1.5
    lambda_auxiliary: float = 0.01
    lambda_lm: float = 1.0

    # lora
    target_modules_lora: List[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    # experts
    num_experts: int = 8
    enable_cache: bool = True
    padding_side: str = "right"

    @staticmethod
    def from_config(config: Dict[str, any]) -> "HotmoeConfig":
        config = HotmoeConfig(**config)
        return config

    def export(self) -> Dict[str, any]:
        config = asdict(self)
        return config
