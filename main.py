#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import logging
from pathlib import Path
import platform

import gradio as gr

import log
from project_settings import environment, project_path, log_directory, time_zone_info

log.setup_size_rotating(log_directory=log_directory, tz_info=time_zone_info)

from tabs.chat_template_tab import get_chat_template_tab
from tabs.shell_tab import get_shell_tab


logger = logging.getLogger("main")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--porter_tasks_file_dir",
        default=(project_path / "data/porter_tasks").as_posix(),
        type=str
    )
    parser.add_argument(
        "--live_recorder_tasks_file",
        default=(project_path / "data/live_recorder_tasks.json").as_posix(),
        type=str
    )
    parser.add_argument(
        "--video_download_tasks_file",
        default=(project_path / "data/video_download_tasks.json").as_posix(),
        type=str
    )
    parser.add_argument(
        "--youtube_video_upload_tasks_file",
        default=(project_path / "data/youtube_video_upload_tasks.json").as_posix(),
        type=str
    )
    parser.add_argument(
        "--bilibili_video_upload_tasks_file",
        default=(project_path / "data/bilibili_video_upload_tasks.json").as_posix(),
        type=str
    )
    parser.add_argument(
        "--live_records_dir",
        default=(project_path / "data/live_records").as_posix(),
        type=str
    )
    parser.add_argument(
        "--server_port",
        default=environment.get("server_port", 7860),
        type=int
    )

    args = parser.parse_args()
    return args



def main():
    args = get_args()

    # ui
    with gr.Blocks() as blocks:
        gr.Markdown(value="live recording.")
        with gr.Tabs():
            _ = get_chat_template_tab()
            _ = get_shell_tab()

    # http://127.0.0.1:7870/
    # http://10.75.27.247:7870/
    blocks.queue().launch(
        # share=True,
        share=False if platform.system() in ("Windows", "Darwin") else False,
        server_name="127.0.0.1" if platform.system() in ("Windows", "Darwin") else "0.0.0.0",
        server_port=args.server_port
    )
    return


if __name__ == "__main__":
    main()
