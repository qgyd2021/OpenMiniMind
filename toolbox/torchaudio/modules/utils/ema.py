#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn


class EMANumpy(object):

    @classmethod
    def _calculate_norm_alpha(cls, sample_rate: int, hop_size: int, tau: float):
        """Exponential decay factor alpha for a given tau (decay window size [s])."""
        dt = hop_size / sample_rate
        result = math.exp(-dt / tau)
        return result

    @classmethod
    def get_norm_alpha(cls, sample_rate: int, hop_size: int, norm_tau: float) -> float:
        a_ = cls._calculate_norm_alpha(sample_rate=sample_rate, hop_size=hop_size, tau=norm_tau)

        precision = 3
        a = 1.0
        while a >= 1.0:
            a = round(a_, precision)
            precision += 1

        return a


class ErbEMA(nn.Module, EMANumpy):
    def __init__(self,
                 sample_rate: int = 8000,
                 hop_size: int = 80,
                 erb_bins: int = 32,
                 mean_norm_init_start: float = -60.,
                 mean_norm_init_end: float = -90.,
                 norm_tau: float = 1.,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.erb_bins = erb_bins
        self.mean_norm_init_start = mean_norm_init_start
        self.mean_norm_init_end = mean_norm_init_end
        self.norm_tau = norm_tau

        self.alpha = self.get_norm_alpha(sample_rate, hop_size, norm_tau)

    def make_erb_norm_state(self) -> torch.Tensor:
        state = torch.linspace(start=self.mean_norm_init_start, end=self.mean_norm_init_end,
                               steps=self.erb_bins)
        state = state.unsqueeze(0).unsqueeze(0)
        # state shape: [b, c, erb_bins]
        # state shape: [1, 1, erb_bins]
        return state

    def norm(self,
             feat_erb: torch.Tensor,
             state: torch.Tensor = None,
             ):
        feat_erb = feat_erb.clone()
        b, c, t, f = feat_erb.shape

        # erb_feat shape: [b, c, t, f]
        if state is None:
            state = self.make_erb_norm_state()
            state = state.to(feat_erb.device)
        state = state.clone()

        for j in range(t):
            current = feat_erb[:, :, j, :]
            new_state = current * (1 - self.alpha) + state * self.alpha

            feat_erb[:, :, j, :] = (current - new_state) / 40.0
            state = new_state

        return feat_erb, state


class SpecEMA(nn.Module, EMANumpy):
    """
    https://github.com/grazder/DeepFilterNet/blob/torchDF_main/libDF/src/lib.rs
    """
    def __init__(self,
                 sample_rate: int = 8000,
                 hop_size: int = 80,
                 df_bins: int = 96,
                 unit_norm_init_start: float = 0.001,
                 unit_norm_init_end: float = 0.0001,
                 norm_tau: float = 1.,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.df_bins = df_bins
        self.unit_norm_init_start = unit_norm_init_start
        self.unit_norm_init_end = unit_norm_init_end
        self.norm_tau = norm_tau

        self.alpha = self.get_norm_alpha(sample_rate, hop_size, norm_tau)

    def make_spec_norm_state(self) -> torch.Tensor:
        state = torch.linspace(start=self.unit_norm_init_start, end=self.unit_norm_init_end,
                               steps=self.df_bins)
        state = state.unsqueeze(0).unsqueeze(0)
        # state shape: [b, c, df_bins]
        # state shape: [1, 1, df_bins]
        return state

    def norm(self,
             feat_spec: torch.Tensor,
             state: torch.Tensor = None,
             ):
        feat_spec = feat_spec.clone()
        b, c, t, f = feat_spec.shape

        # feat_spec shape: [b, 2, t, df_bins]
        if state is None:
            state = self.make_spec_norm_state()
            state = state.to(feat_spec.device)
        state = state.clone()

        for j in range(t):
            current = feat_spec[:, :, j, :]
            current_abs = torch.sum(torch.square(current), dim=1, keepdim=True)
            # current_abs shape: [b, 1, df_bins]
            new_state = current_abs * (1 - self.alpha) + state * self.alpha

            feat_spec[:, :, j, :] = current / torch.sqrt(new_state)
            state = new_state

        return feat_spec, state


MEAN_NORM_INIT = [-60., -90.]


def make_erb_norm_state(erb_bins: int, channels: int) -> np.ndarray:
    state = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], erb_bins)
    state = np.expand_dims(state, axis=0)
    state = np.repeat(state, channels, axis=0)

    # state shape: (audio_channels, erb_bins)
    return state


def erb_normalize(erb_feat: np.ndarray, alpha: float, state: np.ndarray = None):
    erb_feat = np.copy(erb_feat)
    batch_size, time_steps, erb_bins = erb_feat.shape

    if state is None:
        state = make_erb_norm_state(erb_bins, erb_feat.shape[0])
        # state = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], erb_bins)
        # state = np.expand_dims(state, axis=0)
        # state = np.repeat(state, erb_feat.shape[0], axis=0)

    for i in range(batch_size):
        for j in range(time_steps):
            for k in range(erb_bins):
                x = erb_feat[i][j][k]
                s = state[i][k]

                state[i][k] = x * (1. - alpha) + s * alpha
                erb_feat[i][j][k] -= state[i][k]
                erb_feat[i][j][k] /= 40.

    return erb_feat


UNIT_NORM_INIT = [0.001, 0.0001]


def make_spec_norm_state(df_bins: int, channels: int) -> np.ndarray:
    state = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], df_bins)
    state = np.expand_dims(state, axis=0)
    state = np.repeat(state, channels, axis=0)

    # state shape: (audio_channels, df_bins)
    return state


def spec_normalize(spec_feat: np.ndarray, alpha: float, state: np.ndarray = None):
    spec_feat = np.copy(spec_feat)
    batch_size, time_steps, df_bins = spec_feat.shape

    if state is None:
        state = make_spec_norm_state(df_bins, spec_feat.shape[0])

    for i in range(batch_size):
        for j in range(time_steps):
            for k in range(df_bins):
                x = spec_feat[i][j][k]
                s = state[i][k]

                state[i][k] = np.abs(x) * (1. - alpha) + s * alpha
                spec_feat[i][j][k] /= np.sqrt(state[i][k])
    return spec_feat


if __name__ == "__main__":
    pass
