#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
deepspeed --num_gpus=4 step_2_train_model.py
"""
import argparse
import os
from pathlib import Path
import platform

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if platform.system() in ("Windows", "Darwin"):
    from project_settings import project_path
else:
    project_path = os.path.abspath("../../../")
    project_path = Path(project_path)

from peft import LoraConfig
# from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modelscope import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-8B",
        type=str
    ),
    parser.add_argument(
        "--dataset_path",
        default="miyuki2026/tutorials",
        type=str
    ),
    parser.add_argument("--dataset_name", default=None, type=str),
    parser.add_argument("--dataset_split", default=None, type=str),
    parser.add_argument(
        "--dataset_cache_dir",
        # default=(project_path / "hub_datasets").as_posix(),
        default="/root/autodl-tmp/OpenMiniMind/hub_datasets",
        type=str
    ),
    parser.add_argument(
        "--model_cache_dir",
        # default=(project_path / "hub_models").as_posix(),
        default="/root/autodl-tmp/OpenMiniMind/hub_models",
        type=str
    ),
    parser.add_argument("--dataset_streaming", default=None, type=str),
    parser.add_argument("--valid_dataset_size", default=100, type=str),
    parser.add_argument("--shuffle_buffer_size", default=5000, type=str),

    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    ),
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.environ["MODELSCOPE_CACHE"] = args.model_cache_dir

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        quantization_config=None,
        # device_map="auto",
        trust_remote_code=True,
        # cache_dir=args.model_cache_dir,
    )
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        trust_remote_code=True,
        # cache_dir=args.model_cache_dir,
    )
    print(tokenizer)

    def format_func(example):
        formated_text = tokenizer.apply_chat_template(
            example["conversation"],
            tokenize=False,  # 训练时部分词，true返回的是张量
            add_generation_prompt=False,  # 训练期间要关闭，如果是推理则设为True
        )
        return {"formated_text": formated_text}

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
    dataset = dataset_dict["train"]
    print(dataset)

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=args.valid_dataset_size, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    train_dataset = valid_dataset
    train_dataset = train_dataset.map(
        format_func,
        batched=False,
        remove_columns=train_dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # 新写法
        train_dataset=train_dataset,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="formated_text",
            deepspeed="./ds_config/deepspeed_stage_3_config.json",  # 添加deepspeed配置文件
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,  # Use GA to mimic batch size!
            warmup_steps=100,
            num_train_epochs=1,  # Set this for 1 full training run.
            # max_steps = 30,
            learning_rate=3e-5,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0,
            lr_scheduler_type="constant_with_warmup",
            seed=3407,
            report_to="none",  # Use this for WandB etc
        ),
    )

    # 显示当前内存统计信息
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # 显示最终内存和时间统计信息
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # 只保存lora适配器参数
    trained_models_dir = project_path / "trained_models" / "Qwen3-8B-sft-lora-adapter-transformers"
    trained_models_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(trained_models_dir.as_posix())
    tokenizer.save_pretrained(trained_models_dir.as_posix())

    # trained_models_dir = project_path / "trained_models" / "Qwen3-8B-sft-fp16"
    # trained_models_dir.mkdir(parents=True, exist_ok=True)
    # trainer.model.save_pretrained_merged(trained_models_dir.as_posix(), tokenizer, save_method="merged_16bit",)
    # trained_models_dir = project_path / "trained_models" / "Qwen3-8B-sft-int4"
    # trained_models_dir.mkdir(parents=True, exist_ok=True)
    # trainer.model.save_pretrained_merged(trained_models_dir.as_posix(), tokenizer, save_method="merged_4bit",)
    return


if __name__ == "__main__":
    main()
