#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import platform

from huggingface_hub import snapshot_download

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="jingyaogong/MiniMind2", type=str)
    parser.add_argument(
        "--local_dir",
        default=(project_path / "pretrained_models" / "MiniMind2").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # 下载整个仓库
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )

    # 或使用命令行
    # pip install huggingface-hub
    # huggingface-cli download 模型ID --local-dir ./model
    return


if __name__ == "__main__":
    main()
