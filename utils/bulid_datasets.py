import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import datasets
import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset


def format_source(source):
    prefix = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>human\n"
    source = prefix + source + "\n<|im_end|>\n"
    return source


def format_target(target):
    target = "<|im_start|>assistant\n" + target + "<|endoftext|>"
    return target


def build_instruction_dataset_order(
    data_path,
    tokenizer,
    max_seq_length: int,
    data_cache_dir=None,
    data_nums=5000,
    preprocessing_num_workers=None,
    accumulation_steps=5,
):
    def tokenization(examples, task_type):
        sources, targets, task_types = [], [], []

        for idx, (instruction, output) in enumerate(
            zip(examples["instruction"], examples["output"])
        ):
            source = format_source(instruction)
            target = format_target(output)

            sources.append(source)
            targets.append(target)
            task_types.append(f"{task_type}_{idx}")

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(
            targets, return_attention_mask=False, add_special_tokens=False
        )

        all_input_ids, all_labels = [], []
        for s, t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([-100] * len(s) + t)[:max_seq_length]

            assert len(input_ids) == len(labels)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "task_types": task_types,
        }

    logging.warning("Building dataset...")

    num_tasks = len(data_path)
    if accumulation_steps % num_tasks != 0:
        raise ValueError(
            f"accumulation_steps {accumulation_steps}  must be divisible by num_tasks {num_tasks}."
        )

    all_datasets = []

    for task_name, file in data_path.items():
        data_cache_dir = os.path.dirname(file)

        cache_path = os.path.join(
            data_cache_dir,
            f"{os.path.basename(file).split('.')[0]}_{max_seq_length}_{data_nums}",
        )
        os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logging.info(f"Training dataset {file} loaded from disk.")
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            min_data_nums = min(data_nums, len(raw_dataset["train"]))
            subset_dataset = raw_dataset["train"].select(range(min_data_nums))

            tokenization_func = lambda x: tokenization(x, task_name)

            tokenized_dataset = subset_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=subset_dataset.column_names,
                keep_in_memory=False,
                desc="Preprocessing dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)

        processed_dataset.set_format("torch")

        dataset_split = processed_dataset.train_test_split(test_size=0.1, shuffle=False)

        logging.info(
            f"taskname: {task_name}, Train size: {len(dataset_split['train'])}, Test size: {len(dataset_split['test'])}"
        )

        all_datasets.append(dataset_split)

    train_data_per_task = [ds["train"] for ds in all_datasets]
    eval_data_per_task = [ds["test"] for ds in all_datasets]

    max_train_size = max(len(ds) for ds in train_data_per_task)

    mixed_train_data = []
    for i in range(max_train_size):
        for task_id in range(num_tasks):
            if i < len(train_data_per_task[task_id]):
                mixed_train_data.append(train_data_per_task[task_id][i])

    train_dataset = Dataset.from_list(mixed_train_data)
    train_dataset.set_format("torch")
    eval_dataset = concatenate_datasets(eval_data_per_task)

    return train_dataset, eval_dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, task_types = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "task_types")
        )

        # Find max length for padding
        max_length = max([x.size(0) for x in input_ids])

        def left_pad(sequences, pad_value):
            padded = []
            for seq in sequences:
                pad_len = max_length - seq.size(0)
                if pad_len > 0:
                    pad = torch.full((pad_len,), pad_value, dtype=seq.dtype)
                    seq = torch.cat([pad, seq], dim=0)
                padded.append(seq)
            return torch.stack(padded, dim=0)

        input_ids = left_pad(input_ids, self.tokenizer.pad_token_id)
        labels = left_pad(labels, -100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


