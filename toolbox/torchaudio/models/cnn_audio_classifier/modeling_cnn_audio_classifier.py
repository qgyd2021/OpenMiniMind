#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio

from toolbox.torchaudio.models.cnn_audio_classifier.configuration_cnn_audio_classifier import CnnAudioClassifierConfig
from toolbox.torchaudio.configuration_utils import CONFIG_FILE


MODEL_FILE = "model.pt"


name2activation = {
    "relu": nn.ReLU,
}


class Conv1dBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Tuple[int, int],
                 padding: str = 0,
                 dilation: int = 1,
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_channels)
        else:
            self.batch_norm = None

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            stride=stride,
            padding=padding,
            dilation=(dilation,),
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = name2activation[activation]()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: [batch_size, seq_length, spec_dim]
        x = torch.transpose(x, dim0=-1, dim1=-2)

        # x: [batch_size, spec_dim, seq_length]
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = torch.transpose(x, dim0=-1, dim1=-2)
        # x: [batch_size, seq_length, spec_dim]
        return x


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Tuple[int, int],
                 padding: str = 0,
                 dilation: int = 1,
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(in_channels)
        else:
            self.batch_norm = None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding,),
            dilation=(dilation,),
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = name2activation[activation]()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]],
                 dropout: Union[float, List[float]] = 0.0) -> None:

        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise AssertionError("len(hidden_dims) (%d) != num_layers (%d)" %
                                 (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise AssertionError("len(activations) (%d) != num_layers (%d)" %
                                 (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise AssertionError("len(dropout) (%d) != num_layers (%d)" %
                                 (len(dropout), num_layers))
        self._activations = torch.nn.ModuleList([name2activation[activation]() for activation in activations])

        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output


class SpectrogramEncoder(nn.Module):
    def __init__(self,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 ):
        super(SpectrogramEncoder, self).__init__()
        if conv1d_block_param_list is None and conv2d_block_param_list is None:
            raise AssertionError(
                "At least one of the `conv1d_block_param_list` and `conv2d_block_param_list` is required."
            )

        self.conv1d_block_list = None
        if conv1d_block_param_list is not None:
            self.conv1d_block_list = nn.ModuleList(modules=[
                Conv1dBlock(
                    **conv1d_block_param
                )
                for conv1d_block_param in conv1d_block_param_list
            ])

        self.conv2d_block_list = None
        if conv2d_block_param_list is not None:
            self.conv2d_block_list = nn.ModuleList(modules=[
                Conv2dBlock(**conv2d_block_param)
                for conv2d_block_param in conv2d_block_param_list
            ])

    def forward(self,
                inputs: torch.Tensor,
                ):
        # x: [batch_size, spec_dim, seq_length]
        x = inputs

        if self.conv1d_block_list is not None:
            for conv1d_block in self.conv1d_block_list:
                x = conv1d_block(x)

        if self.conv2d_block_list is not None:
            x = torch.unsqueeze(x, dim=1)
            # x: [batch_size, channel, seq_length, spec_dim]
            for conv2d_block in self.conv2d_block_list:
                x = conv2d_block(x)

            # x: [batch_size, channel, seq_length, spec_dim]
            x = torch.transpose(x, dim0=1, dim1=2)
            # x: [batch_size, seq_length, channel, spec_dim]
            batch_size, seq_length, channel, spec_dim = x.shape
            x = torch.reshape(x, shape=(batch_size, seq_length, -1))

        # x: [batch_size, seq_length, spec_dim]
        return x


class WaveEncoder(nn.Module):
    def __init__(self,
                 mel_spectrogram_param: dict,
                 conv1d_block_param_list: List[dict] = None,
                 conv2d_block_param_list: List[dict] = None,
                 ):
        super(WaveEncoder, self).__init__()
        if conv1d_block_param_list is None and conv2d_block_param_list is None:
            raise AssertionError(
                "At least one of the `conv1d_block_param_list` and `conv2d_block_param_list` is required."
            )

        self.wave_to_mel_spectrogram = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=mel_spectrogram_param["sample_rate"],
                n_fft=mel_spectrogram_param["n_fft"],
                win_length=mel_spectrogram_param["win_length"],
                hop_length=mel_spectrogram_param["hop_length"],
                f_min=mel_spectrogram_param["f_min"],
                f_max=mel_spectrogram_param["f_max"],
                window_fn=torch.hamming_window if mel_spectrogram_param["window_fn"] == "hamming" else torch.hann_window,
                n_mels=mel_spectrogram_param["n_mels"],
            ),
        )

        self.spectrogram_encoder = SpectrogramEncoder(
            conv1d_block_param_list=conv1d_block_param_list,
            conv2d_block_param_list=conv2d_block_param_list,
        )

    def forward(self, inputs: torch.Tensor):
        # x: [batch_size, spec_dim, seq_length]
        x = inputs

        with torch.no_grad():
            # shape = [batch_size, spec_dim, seq_length]
            x = self.wave_to_mel_spectrogram(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = x.transpose(1, 2)

        features = self.spectrogram_encoder.forward(x)
        # features: [batch_size, seq_length, spec_dim]
        return features


class ClsHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]],
                 num_labels: int,
                 dropout: Union[float, List[float]] = 0.0
                 ):
        super(ClsHead, self).__init__()

        self.feedforward = FeedForward(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout,
        )

        self.output_project_layer = nn.Linear(self.feedforward.get_output_dim(), num_labels)

    def forward(self, inputs: torch.Tensor):
        # inputs: [batch_size, seq_length, spec_dim]
        x = self.feedforward(inputs)
        # x: [batch_size, seq_length, hidden_size]

        x = torch.mean(x, dim=1)
        # x: [batch_size, hidden_size]

        logits = self.output_project_layer.forward(x)
        # logits: [batch_size, num_labels]
        return logits


class WaveClassifier(nn.Module):
    def __init__(self,
                 wave_encoder: WaveEncoder,
                 cls_head: ClsHead,
                 ):
        super(WaveClassifier, self).__init__()
        self.wave_encoder = wave_encoder
        self.cls_head = cls_head

    def forward(self, inputs: torch.Tensor):
        # x: [batch_size, spec_dim, seq_length]
        x = inputs

        x = self.wave_encoder.forward(x)

        # x: [batch_size, seq_length, spec_dim]
        logits = self.cls_head.forward(x)

        # logits: [batch_size, num_labels]
        return logits


class WaveClassifierPretrainedModel(WaveClassifier):
    def __init__(self,
                 config: CnnAudioClassifierConfig,
                 ):
        super(WaveClassifierPretrainedModel, self).__init__(
            wave_encoder=WaveEncoder(
                mel_spectrogram_param=config.mel_spectrogram_param,
                conv1d_block_param_list=config.conv1d_block_param_list,
                conv2d_block_param_list=config.conv2d_block_param_list,
            ),
            cls_head=ClsHead(
                input_dim=config.cls_head_param["input_dim"],
                num_layers=config.cls_head_param["num_layers"],
                hidden_dims=config.cls_head_param["hidden_dims"],
                activations=config.cls_head_param["activations"],
                num_labels=config.cls_head_param["num_labels"],
                dropout=config.cls_head_param["dropout"],
            )
        )
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = CnnAudioClassifierConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)

        if os.path.isdir(pretrained_model_name_or_path):
            ckpt_file = os.path.join(pretrained_model_name_or_path, MODEL_FILE)
        else:
            ckpt_file = pretrained_model_name_or_path

        with open(ckpt_file, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self,
                        save_directory: Union[str, os.PathLike],
                        state_dict: Optional[dict] = None,
                        ):

        model = self

        if state_dict is None:
            state_dict = model.state_dict()

        os.makedirs(save_directory, exist_ok=True)

        # save state dict
        model_file = os.path.join(save_directory, MODEL_FILE)
        torch.save(state_dict, model_file)

        # save config
        config_file = os.path.join(save_directory, CONFIG_FILE)
        self.config.to_yaml_file(config_file)
        return save_directory


if __name__ == "__main__":
    pass
