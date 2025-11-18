import json
import math
import os
import re
import types
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel, Qwen2Model

from .config import TARGET_MODULE_TYPE, HotmoeConfig


class HotmoeModel:
    @staticmethod
    def from_pretrained(
        model: PreTrainedModel,
        name_or_path: Optional[str] = None,
        eval_mode: bool = True,
    ) -> PeftModel:
        with open(name_or_path + "config.json") as f:
            config = json.load(f)
        config = HotmoeConfig.from_config(config)
        config.torch_dtype = model.dtype
        config.eval_mode = eval_mode
        model = _hotmoe_model(model, config=config)
        
        # Initialize cache: clear all router caches in eval mode
        if eval_mode:
            HotmoeModel._initialize_router_caches(model)
        
        return model
    
    @staticmethod
    def _initialize_router_caches(model):
        for module in model.modules():
            if isinstance(module, AttRouter):
                if module.eval_mode:
                    module.enable_cache()
                    module.clear_cache()  
            elif isinstance(module, Hybrid_router):
                if hasattr(module, 'ee_router') and module.ee_router is not None:
                    if module.ee_router.eval_mode:
                        module.ee_router.enable_cache()
                        module.ee_router.clear_cache()
                if hasattr(module, 'te_router') and module.te_router is not None:
                    if module.te_router.eval_mode:
                        module.te_router.enable_cache()
                        module.te_router.clear_cache()
    
    @staticmethod
    def clear_all_caches(model):
        for module in model.modules():
            if isinstance(module, AttRouter):
                module.clear_cache()
            elif isinstance(module, Hybrid_router):
                if hasattr(module, 'ee_router') and module.ee_router is not None:
                    module.ee_router.clear_cache()
                if hasattr(module, 'te_router') and module.te_router is not None:
                    module.te_router.clear_cache()
    
    @staticmethod
    def enable_all_caches(model):
        for module in model.modules():
            if isinstance(module, AttRouter):
                module.enable_cache()
            elif isinstance(module, Hybrid_router):
                if hasattr(module, 'ee_router') and module.ee_router is not None:
                    module.ee_router.enable_cache()
                if hasattr(module, 'te_router') and module.te_router is not None:
                    module.te_router.enable_cache()
    
    @staticmethod
    def disable_all_caches(model):
        for module in model.modules():
            if isinstance(module, AttRouter):
                module.disable_cache()
            elif isinstance(module, Hybrid_router):
                if hasattr(module, 'ee_router') and module.ee_router is not None:
                    module.ee_router.disable_cache()
                if hasattr(module, 'te_router') and module.te_router is not None:
                    module.te_router.disable_cache()


class Auxloss_fun(nn.Module):
    def __init__(self, config: HotmoeConfig, mix_routers: nn.ModuleList = None):
        super().__init__()
        self.mix_routers = mix_routers
        # loss
        self.use_div_loss = config.use_div_loss

        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm

    def clear(self):
        pass

    def get_auxiliary_loss(self, loss, inputs, reduce="sum", last_minibatch=False):
        auxiliary_loss = []

        for router in self.mix_routers:
            if self.use_div_loss:
                auxiliary_loss.append(
                    router.aux_loss(inputs, last_minibatch=last_minibatch)
                )
            else:
                break

        if len(auxiliary_loss) == 0:
            return loss

        auxiliary_loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == "sum":
            auxiliary_loss = torch.sum(auxiliary_loss)
        elif reduce == "mean":
            auxiliary_loss = torch.mean(auxiliary_loss)
        else:
            raise ValueError(f"reduce must be sum or mean, but got {reduce}")
        loss = self.lambda_lm * loss + self.lambda_auxiliary * auxiliary_loss
        return loss


def similarity_routing_loss(
    avg_hidden_states: torch.Tensor,
    avg_batch_token_routing_weight: torch.Tensor,
    beta_s: float = 1.0,
    beta_d: float = 1.5,
) -> torch.Tensor:
    sim_matrix = F.cosine_similarity(
        avg_hidden_states.unsqueeze(1), avg_hidden_states.unsqueeze(0), dim=-1
    )

    routing_sim_matrix = F.cosine_similarity(
        avg_batch_token_routing_weight.unsqueeze(1),
        avg_batch_token_routing_weight.unsqueeze(0),
        dim=-1,
    )

    s_loss = ((1 - routing_sim_matrix) * sim_matrix).mean()
    d_loss = (routing_sim_matrix * (1 - sim_matrix)).mean()

    sim_route_loss = beta_s * s_loss + beta_d * d_loss

    return sim_route_loss




class AttRouter(nn.Module):
    def __init__(self, config: HotmoeConfig, expert_vector_dim: int):
        super().__init__()
        self.torch_dtype = config.torch_dtype
        self.num_experts: int = config.num_experts
        self.attn_dim: int = config.attn_dim
        self.eval_mode = config.eval_mode

        self.W_q = nn.Linear(
            expert_vector_dim, self.attn_dim, bias=False, dtype=self.torch_dtype
        )  # (output_dim, config.attn_dim)
        self.W_k = nn.Linear(
            expert_vector_dim, self.attn_dim, bias=False, dtype=self.torch_dtype
        )  # (output_dim, config.attn_dim)

        self.routing_weight: Optional[torch.Tensor] = None
        self.batch_token_routing_weight: Optional[torch.Tensor] = None
        self.batch_hidden_states: Optional[torch.Tensor] = None

        self.use_sim_loss = config.use_sim_loss
        self.beta_s = config.beta_s
        self.beta_d = config.beta_d
        
        self.cache_enabled = getattr(config, 'enable_cache', True)  
        self.cached_similarity_matrix: Optional[torch.Tensor] = None
        self.similarity_matrix_cached = False

    def forward(
        self, input_gate, q_expert_vector: torch.Tensor, k_expert_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        inputs:
            expert_vector: (batch, seq_len, num_experts, expert_vector_dim)
            input_gate : (batch, seq_len, num_experts, 1)
        outputs:
            gate: (batch, seq_len, num_experts)
        """

        self.W_q = self.W_q.to(q_expert_vector.device)
        self.W_k = self.W_k.to(k_expert_vector.device)
        
        #  In inference mode, if q_expert_vector and k_expert_vector are the same (EE routing), and the cache is enabled, use the cached similarity matrix
        if (self.cache_enabled and self.eval_mode and 
            torch.equal(q_expert_vector, k_expert_vector) and 
            self.similarity_matrix_cached and 
            self.cached_similarity_matrix is not None and
            self.cached_similarity_matrix.device == q_expert_vector.device):
            

            similarity_matrix = self.cached_similarity_matrix
        else:
            query = self.W_q(q_expert_vector)  # (batch, seq_len, num_experts, 1)
            key = self.W_k(k_expert_vector)  # (batch, seq_len, num_experts, 1)

            similarity_matrix = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
                self.attn_dim
            )  # (batch, seq_len, num_experts, num_experts)
            similarity_matrix = F.softmax(
                similarity_matrix, dim=-1
            )  # (batch, seq_len, num_experts, num_experts)
            
            
            if (self.cache_enabled and self.eval_mode and 
                torch.equal(q_expert_vector, k_expert_vector)):
                self.cached_similarity_matrix = similarity_matrix.detach().clone()
                self.similarity_matrix_cached = True

        gate = torch.matmul(
            similarity_matrix, input_gate
        )  # (batch, seq_len, num_experts, 1)

        self.routing_weight = F.softmax(gate.squeeze(-1), dim=-1)
        return self.routing_weight

    def aux_loss(self, inputs, last_minibatch=False):
        token_routing_weight = self.routing_weight  # (batch, seq_len, num_experts)
        mask = inputs["attention_mask"].to(token_routing_weight.dtype).unsqueeze(-1)
        token_routing_weight = token_routing_weight * mask  # apply mask

        def pad_to_max(tensor, target_length):
            batch_size, seq_len, num_experts = tensor.shape
            if seq_len < target_length:
                pad_size = target_length - seq_len
                tensor = F.pad(tensor, (0, 0, 0, pad_size), mode="constant", value=0)
            return tensor

        if (
            hasattr(self, "batch_token_routing_weight")
            and self.batch_token_routing_weight is not None
        ):
            self.batch_token_routing_weight = self.batch_token_routing_weight.detach()

            current_max_seq_len = max(
                self.batch_token_routing_weight.shape[1], token_routing_weight.shape[1]
            )
            self.batch_token_routing_weight = pad_to_max(
                self.batch_token_routing_weight, current_max_seq_len
            )
            token_routing_weight = pad_to_max(token_routing_weight, current_max_seq_len)

            self.batch_token_routing_weight = torch.cat(
                [self.batch_token_routing_weight, token_routing_weight], dim=0
            )
        else:
            self.batch_token_routing_weight = token_routing_weight

        batch_token_routing_weight = self.batch_token_routing_weight
        full_mask = (batch_token_routing_weight.sum(dim=-1, keepdim=True) > 0).to(
            batch_token_routing_weight.dtype
        )
        num_token = torch.sum(full_mask)

        m = (
            torch.sum(batch_token_routing_weight.view(-1, self.num_experts), dim=0)
            / num_token
        )
        entropy_b = -torch.sum(m * torch.log(m + 1e-9), dim=-1)

        entropy_c = -torch.sum(
            batch_token_routing_weight * torch.log(batch_token_routing_weight + 1e-9),
            dim=-1,
        )
        entropy_c = entropy_c * full_mask.squeeze(-1)
        entropy_c = torch.sum(entropy_c) / num_token

        max_entropy = torch.log(
            torch.tensor(
                self.num_experts,
                dtype=batch_token_routing_weight.dtype,
                device=batch_token_routing_weight.device,
            )
        )
        loss = torch.relu(max_entropy - (entropy_b - entropy_c)) / max_entropy

        if (
            self.use_sim_loss
            and hasattr(self, "batch_hidden_states")
            and self.batch_hidden_states is not None
        ):
            sim_loss = similarity_routing_loss(
                self.batch_hidden_states,
                batch_token_routing_weight.mean(dim=1),
                beta_s=self.beta_s,
                beta_d=self.beta_d,
            ) 


            loss += sim_loss
        if last_minibatch:
            self.batch_token_routing_weight = None
            self.batch_hidden_states = None

        return loss

    def clear(self):
        self.routing_weight = None
    
    def clear_cache(self):
        self.cached_similarity_matrix = None
        self.similarity_matrix_cached = False
    
    def enable_cache(self):
        self.cache_enabled = True
    
    def disable_cache(self):
        self.cache_enabled = False
        self.clear_cache()  


class Hybrid_router(nn.Module):
    def __init__(self, config: HotmoeConfig, input_dim: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_vector_dim = config.expert_vector_dim
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.torch_dtype = config.torch_dtype
        self.top_k = config.top_k
        self.use_sim_loss = config.use_sim_loss
        self.eval_mode = config.eval_mode

        self.use_EE_router = config.use_EE_router
        self.use_TE_router = config.use_TE_router
        self.mlp_rank = 4

        self.expert_embedding = nn.Embedding(
            self.num_experts, self.expert_vector_dim, dtype=self.torch_dtype
        )
        self.layernorm = nn.LayerNorm(
            [
                self.num_experts,
                self.expert_vector_dim,
            ],
            dtype=self.torch_dtype,
        )

        self.W_v = nn.Linear(
            input_dim, self.num_experts, dtype=self.torch_dtype
        )  # (input_dim, num_experts)
        if self.use_EE_router:
            self.ee_router = AttRouter(config, self.expert_vector_dim)
        if self.use_TE_router:
            self.mlp = nn.Sequential(
                nn.Dropout(p=config.dropout),
                nn.Linear(input_dim, self.mlp_rank, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(
                    self.mlp_rank,
                    self.num_experts * self.expert_vector_dim,
                    dtype=torch.float32,
                ),
            )
            self.te_router = AttRouter(config, self.expert_vector_dim)

        self.routing_weight = None

    def _pad_to_max(self, tensor, target_length):
        batch_size, seq_len, num_experts = tensor.shape
        if seq_len < target_length:
            pad_size = target_length - seq_len
            tensor = F.pad(tensor, (0, 0, 0, pad_size), mode="constant", value=0)
        return tensor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global_experts = self.expert_embedding.weight.unsqueeze(0).unsqueeze(0)
        global_experts = self.layernorm(global_experts)
        input_gate = self.W_v(hidden_states).unsqueeze(
            -1
        )  # (batch, seq_len, num_experts, 1)

        if self.use_EE_router:
            gate_EE = self.ee_router(input_gate, global_experts, global_experts)

        if self.use_TE_router:
            input_dtype = hidden_states.dtype
            x = hidden_states.to(torch.float32)

            batch_size, seq_len, input_dim = hidden_states.shape
            x = self.mlp.float()(x)
            x = x.to(input_dtype)
            dyn_experts = x.view(
                batch_size, seq_len, self.num_experts, self.expert_vector_dim
            )
            dyn_experts = self.layernorm(dyn_experts)
            gate_TE = self.te_router(input_gate, dyn_experts, global_experts)

        if self.use_TE_router:
            routing_weight = gate_TE
        elif self.use_EE_router:
            routing_weight = gate_EE
        else:
            routing_weight = F.softmax(input_gate.squeeze(-1), dim=-1)

        if self.top_k_routing_strategy:
            top_k_values, top_k_indices = torch.topk(routing_weight, self.top_k, dim=-1)
            routing_weight = torch.full_like(
                routing_weight, torch.finfo(routing_weight.dtype).min
            )
            routing_weight.scatter_(-1, top_k_indices, top_k_values)
            routing_weight = torch.softmax(routing_weight / 0.1, dim=-1)
        self.routing_weight = routing_weight

        # only for training with sim_loss

        # seq level
        if self.use_sim_loss and not self.eval_mode:
            hidden_mean = hidden_states.mean(dim=1).detach()
            if self.use_EE_router:
                self.ee_router.routing_weight = routing_weight
                if self.ee_router.batch_hidden_states is not None:
                    self.ee_router.batch_hidden_states = torch.cat(
                        (self.ee_router.batch_hidden_states, hidden_mean), dim=0
                    )
                else:
                    self.ee_router.batch_hidden_states = hidden_mean
            if self.use_TE_router:
                self.te_router.routing_weight = routing_weight
                if self.te_router.batch_hidden_states is not None:
                    self.te_router.batch_hidden_states = torch.cat(
                        (self.te_router.batch_hidden_states, hidden_mean), dim=0
                    )
                else:
                    self.te_router.batch_hidden_states = hidden_mean


        return routing_weight

    def aux_loss(self, attention_mask, last_minibatch=False):
        loss = torch.tensor(0.0, device=self.routing_weight.device)
        if self.use_TE_router:
            loss += self.te_router.aux_loss(attention_mask, last_minibatch)
        elif self.use_EE_router:
            loss += self.ee_router.aux_loss(attention_mask, last_minibatch)

        return loss.to(self.routing_weight.dtype)

    def get_routing_weight(self):
        return self.routing_weight


class MoRa(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: HotmoeConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype

        self.num_experts = config.num_experts
        self.rank = config.lora_r
        self.dropout = nn.Dropout(p=config.dropout)
        
        self.mora_a = nn.Parameter(
            torch.empty(
                (self.rank * self.num_experts, self.in_features), dtype=self.dtype_
            )
        )
        self.mora_b = nn.Parameter(
            torch.empty(
                (self.out_features, self.rank * self.num_experts), dtype=self.dtype_
            )
        )
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mora_a, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b)

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.mora_a)

        target_shape = hidden_states.shape[:-1] + (self.num_experts, self.rank)
        hidden_states = hidden_states.view(target_shape)

        hidden_states = (hidden_states * gate.unsqueeze(-1)).view(
            hidden_states.shape[:-2] + (-1,)
        )
        hidden_states = F.linear(hidden_states, self.mora_b) * self.scaling
        return hidden_states.to(residual.dtype) + residual


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: HotmoeConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.rank = config.lora_r
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), dtype=self.dtype_)
        )
        self.lora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank), dtype=self.dtype_)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.lora_a)
        hidden_states = F.linear(hidden_states, self.lora_b) * self.scaling
        return hidden_states


class AdapterLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        config: HotmoeConfig,
        router: Hybrid_router,
        use_lora: bool = False,
    ):
        super().__init__()
        # linear
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.weight = base_layer.weight
        if hasattr(base_layer, "bias"):
            self.bias = base_layer.bias
        else:
            self.register_parameter("bias", None)

        self.use_lora = use_lora

        # mixture of lora experts
        if self.use_lora:
            self.lora = LoRA(base_layer, config)
        else:
            self.mora = MoRa(base_layer, config)
        # routers
        self.router = router

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.lora(hidden_states) + result
        else:
            gate = self.router(hidden_states)  
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.mora(hidden_states, gate, result)


def get_peft_model(model: PreTrainedModel, config: HotmoeConfig) -> PeftModel:
    config.hidden_size = model.config.hidden_size
    config.model_type = model.config.model_type
    model = _hotmoe_model(model, config)
    return model


def _get_module(model: nn.Module, target_name: str):
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None


def _choose_router(layer_id, config):
    ratio = (layer_id + 1) / (config.max_llm_layer + 1)

    router_mapping = {
        0.5: {"ee": True, "te": False},
        1.0: {"ee": False, "te": True},
    }
    for threshold, router in router_mapping.items():
        if ratio <= threshold:
            config.use_EE_router = router["ee"]
            config.use_TE_router = router["te"]
            break


def _apply_for_layer(layer_module: nn.Module, layer_id: int, config: HotmoeConfig):
    hybrid_router_list = []

    def get_hybrid_router(input_dim, tag: Optional[str] = None):
        router = Hybrid_router(config, input_dim)
        hybrid_router_list.append(router)
        return router

    def get_target_modules(target: list[str]) -> list[str]:
        res = []
        for t in target:
            res += TARGET_MODULE_TYPE[config.model_type][t]
        return res

    def set_hotmoe(module, targets: list):
        hybrid_router = None
        for target_name in targets:
            if target_name not in config.target_modules:
                continue
            target_model = _get_module(module, target_name)
            if not isinstance(target_model, nn.Linear):
                continue
            if (
                config.target_modules_lora is not None
                and target_name in config.target_modules_lora
            ):
                target_model = AdapterLinear(
                    target_model, config, router=None, use_lora=True
                )
            else:
                _choose_router(layer_id, config)  # adjust config for each layer
                if hybrid_router is None:
                    hybrid_router = get_hybrid_router(
                        target_model.in_features, tag=target_name
                    )
                target_model = AdapterLinear(
                    target_model, config, router=hybrid_router, use_lora=False
                )
            setattr(module, target_name, target_model)

    # apply for attention block
    atte_name = TARGET_MODULE_TYPE[config.model_type]["atte"]
    atte_module = _get_module(layer_module, atte_name)

    target_modules = get_target_modules(["q", "k", "v", "o"])
    for target_module in target_modules:
        set_hotmoe(atte_module, [target_module])

    # apply for ffn block
    ffn_name = TARGET_MODULE_TYPE[config.model_type]["ffn"]
    ffn_module = _get_module(layer_module, ffn_name)
    target_modules = get_target_modules(["wi", "wo"])
    for target_module in target_modules:
        set_hotmoe(ffn_module, [target_module])

    return hybrid_router_list


def _hotmoe_model(model, config: HotmoeConfig) -> PeftModel:
    def _get_layer_id(name: str):
        match = re.search(r"\.\d+$", name)
        if match:
            return int(name.split(".")[-1])
        return None

    layer_list = dict()  # {layer_id : layer_module}
    for module_name, module in model.named_modules():
        layer_id = _get_layer_id(module_name)
        if layer_id is not None:
            if layer_id > config.max_llm_layer:
                # record the max layer id
                config.max_llm_layer = layer_id
            # record decoder layers
            layer_list[layer_id] = module

    hybrid_router_list = nn.ModuleList()
    for layer_id in sorted(layer_list.keys()):
        module = layer_list[layer_id]
        hybrid_routers = _apply_for_layer(module, layer_id, config)
        for mix_router in hybrid_routers:
            hybrid_router_list.append(mix_router)

    # loss manager
    auxloss_functions = Auxloss_fun(config, hybrid_router_list)
    model.auxloss_functions = auxloss_functions

    trainable_modules = [
        "router",
        "mora",
        "lora",
    ]
    # freeze parameters
    for param_name, param in model.named_parameters():
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # overwrite save_pretrained
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    # model.peft_config = config
    setattr(model, "peft_config", config)
    return model


def _save_pretrained(self: nn.Module, path, **kwargs):
    if not os.path.exists(path):
        os.makedirs(path)
    trainable_params = dict()
    for name, param in self.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach().cpu()
    config = self.peft_config.export()
    torch.save(trainable_params, path + "/" + "adapter_model.safetensors")
    config["torch_dtype"] = None
    json.dump(config, open(path + "/" + "config.json", "w"))
