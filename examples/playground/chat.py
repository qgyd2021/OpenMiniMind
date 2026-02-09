#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/jingyaogong/minimind/blob/master/eval_llm.py
"""
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        # default="jingyaogong/MiniMind2",
        default=(project_path / "pretrained_models/MiniMind2"),
        type=str
    )

    parser.add_argument(
        "--max_new_tokens",
        default=8192, # 8192, 128
        type=int, help="最大生成长度（注意：并非模型实际长文本能力）"
    )
    parser.add_argument("--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument("--temperature", default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")

    parser.add_argument(
        "--show_speed",
        default=1,  # 1, 0
        type=int, help="显示decode速度（tokens/s）"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # device = "mps"
        device = "cpu"
    else:
        device = "cpu"
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    model = model.eval().to(device)
    # print(tokenizer)
    # print(model)

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食"
    ]
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # conversation = list()
    conversation = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]
    while True:
        if input_mode == 0:
            if len(prompts) == 0:
                break
            user_input = prompts.pop(0)
            print(f"💬: {user_input}")
        else:
            user_input = input("💬: ")
            user_input = str(user_input).strip()
        conversation.append({"role": "user", "content": user_input})
        inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer.__call__(
            inputs,
            return_tensors="pt",
            truncation=True
        )
        inputs = inputs.to(device)
        # print(inputs)

        print("🤖: ", end="")
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0,
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f"\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n") if args.show_speed else print("\n\n")

    return


if __name__ == "__main__":
    main()
