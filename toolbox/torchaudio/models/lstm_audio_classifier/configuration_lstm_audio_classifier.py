#!/usr/bin/python3
# -*- coding: utf-8 -*-
from toolbox.torchaudio.configuration_utils import PretrainedConfig


class WaveClassifierConfig(PretrainedConfig):
    def __init__(self,
                 mel_spectrogram_param: dict,
                 lstm_layer_param: dict,
                 pooling_layer_param: dict,
                 cls_head_param: dict,
                 **kwargs
                 ):
        super(WaveClassifierConfig, self).__init__(**kwargs)
        self.mel_spectrogram_param = mel_spectrogram_param
        self.lstm_layer_param = lstm_layer_param
        self.pooling_layer_param = pooling_layer_param
        self.cls_head_param = cls_head_param


if __name__ == "__main__":
    pass
