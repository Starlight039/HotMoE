import json
import logging
import transformers
from torch.optim import AdamW
from transformers import AutoTokenizer, HfArgumentParser, get_scheduler

from hotmoe import HotMoEDataArguments, HotMoEModelArguments, HotMoETrainingArguments
from mod_transformers.trainer import Hotmoe_Trainer
from utils.bulid_datasets import (
    DataCollatorForSupervisedDataset,
    build_instruction_dataset_order,
)
from utils.function import save_check_point, set_seed
from utils.set_config import setup

transformers.logging.set_verbosity_error()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_scheduler(optimizer, args):
    if args.lr_scheduler_type == "polynomial":
        specific_kwargs = {"power", args.polynomial_power}
    elif args.lr_scheduler_type == "cosine_with_restarts":
        specific_kwargs = {"num_cycles", args.num_cycles}
    else:
        specific_kwargs = None

    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_training_steps,
        scheduler_specific_kwargs=specific_kwargs,
    )
    return scheduler


def main():
    parser = HfArgumentParser(
        (HotMoEModelArguments, HotMoEDataArguments, HotMoETrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model)
    model = setup(model_args, tokenizer)

    # Load dataset
    with open(data_args.dataset_json_path, "r", encoding="utf-8") as f:
        dataset_paths = json.load(f)

    train_dataset, val_dataset = build_instruction_dataset_order(
        dataset_paths,
        tokenizer,
        max_seq_length=data_args.max_seq_length,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        data_nums=data_args.data_nums,
        accumulation_steps=training_args.gradient_accumulation_steps,
    )
    logging.info(f"Train dataset size:{len(train_dataset)}")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    scheduler = create_scheduler(optimizer, training_args)

    trainer = Hotmoe_Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()
    save_check_point(trainer.model, training_args, tokenizer)


if __name__ == "__main__":
    main()
