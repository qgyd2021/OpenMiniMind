#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import platform

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

if platform.system() in ("Windows", "Darwin"):
    from project_settings import project_path
else:
    project_path = os.path.abspath("../../../")
    project_path = Path(project_path)

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        type=str
    )
    parser.add_argument(
        "--lora_adapter_path",
        default=(project_path / "trained_models" / "Qwen3-8B-sft-lora-adapter-unsloth").as_posix(),
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
    parser.add_argument("--valid_dataset_size", default=1000, type=str),
    parser.add_argument("--shuffle_buffer_size", default=5000, type=str),

    parser.add_argument(
        "--max_new_tokens",
        default=1024, # 8192, 128
        type=int, help="最大生成长度（注意：并非模型实际长文本能力）"
    )
    parser.add_argument("--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument("--temperature", default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")

    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    )
    parser.add_argument("--output_file", default="evaluation.jsonl", type=str),

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,  # 支持32K+长上下文
        device_map="auto",
        dtype=None,  # 自动选择最优精度
        load_in_4bit=True,  # 4bit量化节省70%显存
    )

    # 2、注入lora适配器
    model.load_adapter(args.lora_adapter_path)

    # 启用unsloth推理加速
    FastLanguageModel.for_inference(model)
    model.eval()

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
        # train_dataset = dataset.skip(args.valid_dataset_size)
        # train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=args.valid_dataset_size, seed=None)
        # train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    with open(output_file.as_posix(), "w", encoding="utf-8") as f:
        for example in tqdm(valid_dataset):
            conversation = example["conversation"]
            prompt = conversation[:-1]
            response = conversation[-1]["content"]

            format_messages = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,  # 训练时部分词，true返回的是张量
                add_generation_prompt=True,  # 训练期间要关闭，如果是推理则设为True
            )

            # 4、调用tokenizer得到input
            inputs = tokenizer(format_messages, return_tensors="pt").to(model.device)

            # 5、调用model.generate()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens, do_sample=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0,
            )

            response_: str = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            response_ = response_.split("</think>")[-1].strip()

            row = {
                "prompt": prompt,
                "response": response,
                "response_": response_,
            }
            row = json.dumps(row, ensure_ascii=False)
            f.write(f"{row}\n")
    return


if __name__ == "__main__":
    main()
