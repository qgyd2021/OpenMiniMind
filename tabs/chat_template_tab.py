#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json

import gradio as gr
from transformers import AutoTokenizer


def run_chat_template(conversation: str, model_name: str, add_generation_prompt: bool = False):
    conversation = json.loads(conversation)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    result = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return result


def get_chat_template_tab():
    with gr.TabItem("chat_template"):
        model_name_choices = [
            "Qwen/Qwen3-8B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "openai/gpt-oss-20b",
            "jingyaogong/MiniMind2",

        ]
        ct_model_name = gr.Dropdown(choices=model_name_choices, value=model_name_choices[0], label="model_name")
        ct_conversation = gr.Textbox(label="conversation")
        ct_add_generation_prompt = gr.Checkbox(label="add_generation_prompt")
        ct_tokenize = gr.Button("tokenize")
        ct_output = gr.Textbox(label="output", max_lines=100)

        ct_tokenize.click(
            run_chat_template,
            inputs=[ct_conversation, ct_model_name, ct_add_generation_prompt],
            outputs=[ct_output],
        )

        gr.Examples(
            examples=[
                [
                    json.dumps([{"role": "user", "content": "帮我识别出文本中的关键词：\n凉山彝族社会中的\"尔普\"(份子钱)是一种礼物交换形式.对\"尔普\"的研究和分析,可有助于人们理解凉山彝族社会.\"尔普\"本来是维系彝族传统社会宗族内部亲属组织的纽带,由于文化变迁的原因,后来发展出了跨宗族的\"尔普\"新形式,又由于族群互动的原因,还产生了跨越族群的\"尔普\"形式.\"尔普\"形式的变迁是族群互动下的一种文化变迁形式,其动力来源于彝、汉两族的互动关系.彝族社会中\"尔普\"的变迁形式是人类学关于族群互动下的文化变迁理论的鲜活事例."}, {"role": "assistant", "content": "彝族;尔普;礼物交换;族群互动"}], ensure_ascii=False),
                    "Qwen/Qwen3-8B",
                    True,
                ]
            ],
            inputs=[ct_conversation, ct_model_name, ct_add_generation_prompt],
            outputs=[ct_output],
            fn=run_chat_template,
        )

    return locals()


if __name__ == "__main__":
    pass
