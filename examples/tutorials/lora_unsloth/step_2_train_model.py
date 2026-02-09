#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import platform

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

if platform.system() in ("Windows", "Darwin"):
    from project_settings import project_path
else:
    project_path = os.path.abspath("../../../")
    project_path = Path(project_path)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        type=str
    )
    parser.add_argument(
        "--dataset_path",
        default="miyuki2026/tutorials",
        type=str
    ),
    parser.add_argument("--dataset_name", default=None, type=str),
    parser.add_argument("--dataset_split", default=None, type=str),
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    ),
    parser.add_argument("--dataset_streaming", default=None, type=str),

    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    ),
    args = parser.parse_args()
    return args


def convert_to_qwen_format(example):
    """
    :param example: {"conversation_id": 612, "category": "", "conversation": [{"human": "", "assistant": ""}], "dataset": ""}
    """
    conversation_ = []
    for conversation in example["conversation"]:
        conversation_.append([
            {"role": "user", "content": conversation["human"].strip()},
            {"role": "assistant", "content": conversation["assistant"].strip()},
        ])
    result = {"conversation": conversation_}
    return result


def main():
    args = get_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        device_map="auto",
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=32,  # Best to choose alpha = rank or rank*2
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # rank stabilized LoRA
        loftq_config=None,  # LoftQ
    )
    print(model)

    def format_func(example):
        formated_text = tokenizer.apply_chat_template(
            example["conversation"],
            tokenize=False,  # 训练时部分词，true返回的是张量
            add_generation_prompt=False,  # 训练期间要关闭，如果是推理则设为True
        )
        return {"text": formated_text}

    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        data_dir="keywords",
        # data_dir="psychology",
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        # num_proc=args.num_workers if not args.dataset_streaming else None,
        streaming=args.dataset_streaming,
    )
    print(dataset_dict)
    train_dataset = dataset_dict["train"]

    train_dataset = train_dataset.map(
        convert_to_qwen_format,
        batched=False,
        remove_columns=["conversation_id", "category", "dataset"]
    )
    print(train_dataset)

    train_dataset = train_dataset.map(
        format_func,
        batched=False,
        remove_columns=["conversation_id", "category", "dataset"]
    )
    print(train_dataset)

    return


if __name__ == "__main__":
    main()
