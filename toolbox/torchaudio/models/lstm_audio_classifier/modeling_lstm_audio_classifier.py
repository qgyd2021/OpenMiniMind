#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torchaudio
import torch.nn as nn
from toolbox.torchaudio.configuration_utils import CONFIG_FILE, PretrainedConfig
from toolbox.torchaudio.models.lstm_audio_classifier.configuration_lstm_audio_classifier import WaveClassifierConfig
from toolbox.torchaudio.modules.conv_stft import ConvSTFT
from toolbox.torchaudio.modules.freq_bands.mel_bands import MelBands


MODEL_FILE = "model.pt"


name2activation = {
    "relu": nn.ReLU,
}


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


class PoolingLayer(nn.Module):
    def __init__(self,
                 pool_layer: str,
                 ):
        super(PoolingLayer, self).__init__()
        # mean, last
        self.pool_layer = pool_layer

    def forward(self, inputs: torch.Tensor):
        # inputs shape: [b, t, f]
        if self.pool_layer == "mean":
           inputs = torch.mean(inputs, dim=1)
        elif self.pool_layer == "last":
            inputs = inputs[:, -1, :]
        else:
            raise ValueError("pool_layer must be mean or last")
        # inputs shape: [b, f]
        return inputs


class LSTMLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 ):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

    def forward(self, inputs: torch.Tensor, h: Optional[torch.Tensor] = None, c: Optional[torch.Tensor] = None):
        """
        :param inputs: shape, [b, t, f]
        :param h: shape, [num_layers, b, hidden_size]
        :param c: shape, [num_layers, b, hidden_size]
        :return:
            features: shape, [b, hidden_size]
            h: shape, [num_layers, b, hidden_size]
            c: shape, [num_layers, b, hidden_size]
        """
        if h is None or c is None:
            batch_size = inputs.size(0)
            h, c = self._init_hidden(batch_size, inputs.device)
        if inputs.dim() == 4:
            # [b, 1, t, f]
            inputs = inputs.squeeze(1)
            # [b, t, f]

        # [b, t, f]
        features, (h, c) = self.lstm(inputs, (h, c))
        return features, h, c

    def _init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0


class WaveEncoder(nn.Module):
    def __init__(self,
                 mel_spectrogram_param: dict,
                 lstm_layer_param: dict,
                 ):
        super().__init__()
        self.mel_spectrogram_param = mel_spectrogram_param
        self.lstm_layer_param = lstm_layer_param

        self.stft = ConvSTFT(
            nfft=mel_spectrogram_param["n_fft"],
            win_size=mel_spectrogram_param["win_length"],
            hop_size=mel_spectrogram_param["hop_length"],
            win_type=mel_spectrogram_param["window_fn"],
            power=True,
            requires_grad=False,
        )
        self.mel_scale = MelBands(
            n_fft=mel_spectrogram_param["n_fft"],
            n_mels=mel_spectrogram_param["n_mels"],
            sample_rate=mel_spectrogram_param["sample_rate"],
            f_min=mel_spectrogram_param["f_min"],
            f_max=mel_spectrogram_param["f_max"],
        )
        self.lstm_layer = LSTMLayer(
            input_size=lstm_layer_param["input_size"],
            hidden_size=lstm_layer_param["hidden_size"],
            num_layers=lstm_layer_param["num_layers"],
            dropout=lstm_layer_param["dropout"],
        )

    def forward(self, inputs: torch.Tensor):
        # x: [b, num_samples]
        x = inputs

        with torch.no_grad():
            x = self.stft.forward(x)
            # shape = [b, f, t]
            x = x.transpose(1, 2)
            # shape = [b, t, f]
            x = self.mel_scale.mel_scale(x)
            # shape = [b, t, mel_bins]
            x = x + 1e-6
            x = x.log()

        # x shape = [b, t, mel_bins]
        features, h, c = self.lstm_layer.forward(x)
        # features: shape, [b, t, hidden_size]
        # h: shape, [num_layers, b, hidden_size]
        # c: shape, [num_layers, b, hidden_size]
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
        # inputs: [b, f]
        inputs = torch.unsqueeze(inputs, dim=1)
        # inputs: [b, 1, f]
        x = self.feedforward(inputs)
        # inputs: [b, 1, f]
        x = torch.squeeze(x, dim=1)
        # x: [b, f]

        logits = self.output_project_layer.forward(x)
        # logits: [b, num_labels]
        return logits


class WaveClassifier(nn.Module):
    def __init__(self,
                 wave_encoder: WaveEncoder,
                 pooling_layer: PoolingLayer,
                 cls_head: ClsHead
                 ):
        super(WaveClassifier, self).__init__()
        self.wave_encoder = wave_encoder
        self.pooling_layer = pooling_layer
        self.cls_head = cls_head

    def forward(self,
                inputs: torch.Tensor,
                ):
        # inputs shape: [b, num_samples]
        features = self.wave_encoder.forward(inputs)
        # features shape: [b, t, f]
        feature = self.pooling_layer.forward(features)
        # features shape: [b, f]
        logits = self.cls_head.forward(feature)
        # logits shape: [batch_size, num_classes]
        return logits


class WaveClassifierPretrainedModel(WaveClassifier):
    def __init__(self,
                 config: WaveClassifierConfig,
                 ):
        super(WaveClassifierPretrainedModel, self).__init__(
            wave_encoder=WaveEncoder(
                mel_spectrogram_param=config.mel_spectrogram_param,
                lstm_layer_param=config.lstm_layer_param,
            ),
            pooling_layer=PoolingLayer(
                pool_layer=config.pooling_layer_param["pool_layer"],
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
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config = WaveClassifierConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

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


class WaveClassifierExport(WaveClassifierPretrainedModel):
    def __init__(self, config: WaveClassifierConfig):
        super(WaveClassifierExport, self).__init__(config=config)

    def forward(self,
                inputs: torch.Tensor,
                h: torch.Tensor = None,
                c: torch.Tensor = None,
                ):
        # x: [b, num_samples]
        x = inputs

        with torch.no_grad():
            x = self.wave_encoder.stft.forward(x)
            # shape = [b, freq_bins, t]
            x = x.transpose(1, 2)
            # shape = [b, t, freq_bins]
            x = self.wave_encoder.mel_scale.mel_scale(x)
            # shape = [b, t, mel_bins]
            spec = x + 1e-6
            spec = spec.log()
        # spec shape = [b, t, f]
        features, h, c = self.wave_encoder.lstm_layer.forward(spec, h=h, c=c)
        # features: shape, [b, t, hidden_size]
        # h: shape, [num_layers, b, hidden_size]
        # c: shape, [num_layers, b, hidden_size]

        # features shape: [b, t, f]
        feature = self.pooling_layer.forward(features)
        # features shape: [b, f]
        logits = self.cls_head.forward(feature)
        # logits shape: [batch_size, num_classes]
        return logits, h, c


def main():
    config = WaveClassifierConfig.from_pretrained("examples/lstm_classifier.yaml")
    model = WaveClassifierPretrainedModel(config)
    model_export = WaveClassifierExport(config)
    model.eval()
    model_export.eval()

    inputs = torch.rand(size=(1, 16000), dtype=torch.float32)

    logits = model.forward(inputs)
    print(logits)

    logits, h, c = model_export.forward(inputs)

    return


if __name__ == "__main__":
    main()
