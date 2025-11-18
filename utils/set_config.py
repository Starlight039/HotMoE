import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hotmoe import HotmoeConfig, get_peft_model

DEFAULT_TARGET_MODULES = {
    "qwen2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
}


def setup(args, tokenizer):
    # base model
    model_config = AutoConfig.from_pretrained(args.model)

    if args.use_deepspeed:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"

    if args.target_modules is None:
        args.target_modules = DEFAULT_TARGET_MODULES[model_config.model_type]

    # number of parameters of base model
    vanilla_params = sum(p.numel() for p in model.parameters())

    peft_config = HotmoeConfig(
        target_modules=args.target_modules,
        dropout=args.dropout,
        # routing strategy
        top_k_routing_strategy=args.top_k_routing_strategy,
        top_k=args.top_k,
        attn_dim=args.attn_dim,
        expert_vector_dim=args.expert_vector_dim,
        # loss
        use_div_loss=args.use_div_loss,
        use_sim_loss=args.use_sim_loss,
        lambda_lm=args.lambda_lm,
        lambda_auxiliary=args.lambda_auxiliary,
        beta_s=args.beta_s,
        beta_d=args.beta_d,
        # experts
        num_experts=args.num_experts,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    peft_config.torch_dtype = torch.float32
    peft_config.padding_side = tokenizer.padding_side
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        "model parameters :",
        vanilla_params,
        " | trainable parameters :",
        trainable_params,
        " | rate :",
        trainable_params / vanilla_params,
    )

    return model
