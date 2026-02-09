#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import platform

if platform.system() in ("Windows", "Darwin"):
    from project_settings import project_path
else:
    project_path = os.path.abspath("../../../")
    project_path = Path(project_path)

from unsloth import FastLanguageModel
from transformers import TextStreamer


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
    ),
    args = parser.parse_args()
    return args


def main():
    args = get_args()

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

    messages = [
        {
            "role": "user",
            "content": "关键词识别：\n梯度功能材料是基于一种全新的材料设计概念而开发的新型功能材料.陶瓷-金属FGM的主要结构特点是各梯度层由不同体积浓度的陶瓷和金属组成,材料在升温和降温过程中宏观梯度层间产生热应力,每一梯度层中细观增强相和基体的热物性失配将产生单层热应力,从而导致材料整体的破坏.采用云纹干涉法,对具有四个梯度层的SiC/A1梯度功能材料分别在机载、热载及两者共同作用下进行了应变测试,分别得到了这三种情况下每梯度层同一位置的纵向应变,横向应变和剪应变值."
        }
    ]
    format_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 训练时部分词，true返回的是张量
        add_generation_prompt=True,  # 训练期间要关闭，如果是推理则设为True
    )

    # 4、调用tokenizer得到input
    inputs = tokenizer(format_messages, return_tensors="pt").to(model.device)

    # 5、调用model.generate()
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens, do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只输出回答部分

    print(response)
    return


if __name__ == "__main__":
    main()
