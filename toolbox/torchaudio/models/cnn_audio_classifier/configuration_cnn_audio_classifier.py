#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Union

from toolbox.torchaudio.configuration_utils import PretrainedConfig


class CnnAudioClassifierConfig(PretrainedConfig):
    def __init__(self,
                 mel_spectrogram_param: dict,
                 cls_head_param: dict,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 **kwargs
                 ):
        super(CnnAudioClassifierConfig, self).__init__(**kwargs)
        self.mel_spectrogram_param = mel_spectrogram_param
        self.cls_head_param = cls_head_param
        self.conv1d_block_param_list = conv1d_block_param_list
        self.conv2d_block_param_list = conv2d_block_param_list


if __name__ == "__main__":
    pass
