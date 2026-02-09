#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn


class ErbBandsNumpy(object):

    @staticmethod
    def freq2erb(freq_hz: float) -> float:
        """
        https://www.cnblogs.com/LXP-Never/p/16011229.html
        1 / (24.7 * 9.265) = 0.00436976
        """
        return 9.265 * math.log(freq_hz / (24.7 * 9.265) + 1)

    @staticmethod
    def erb2freq(n_erb: float) -> float:
        return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1)

    @classmethod
    def get_erb_widths(cls, sample_rate: int, nfft: int, erb_bins: int, min_freq_bins_for_erb: int) -> np.ndarray:
        """
        https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs
        :param sample_rate:
        :param nfft:
        :param erb_bins: erb (Equivalent Rectangular Bandwidth) 等效矩形带宽的通道数.
        :param min_freq_bins_for_erb: Minimum number of frequency bands per erb band
        :return:
        """
        nyq_freq = sample_rate / 2.
        freq_width: float = sample_rate / nfft

        min_erb: float = cls.freq2erb(0.)
        max_erb: float = cls.freq2erb(nyq_freq)

        erb = [0] * erb_bins
        step = (max_erb - min_erb) / erb_bins

        prev_freq_bin = 0
        freq_over = 0
        for i in range(1, erb_bins + 1):
            f = cls.erb2freq(min_erb + i * step)
            freq_bin = int(round(f / freq_width))
            freq_bins = freq_bin - prev_freq_bin - freq_over

            if freq_bins < min_freq_bins_for_erb:
                freq_over = min_freq_bins_for_erb - freq_bins
                freq_bins = min_freq_bins_for_erb
            else:
                freq_over = 0
            erb[i - 1] = freq_bins
            prev_freq_bin = freq_bin

        erb[erb_bins - 1] += 1
        too_large = sum(erb) - (nfft / 2 + 1)
        if too_large > 0:
            erb[erb_bins - 1] -= too_large
        return np.array(erb, dtype=np.uint64)

    @staticmethod
    def get_erb_filter_bank(erb_widths: np.ndarray,
                            normalized: bool = True,
                            inverse: bool = False,
                            ):
        num_freq_bins = int(np.sum(erb_widths))
        num_erb_bins = len(erb_widths)

        fb: np.ndarray = np.zeros(shape=(num_freq_bins, num_erb_bins))

        points = np.cumsum([0] + erb_widths.tolist()).astype(int)[:-1]
        for i, (b, w) in enumerate(zip(points.tolist(), erb_widths.tolist())):
            fb[b: b + w, i] = 1

        if inverse:
            fb = fb.T
            if not normalized:
                fb /= np.sum(fb, axis=1, keepdims=True)
        else:
            if normalized:
                fb /= np.sum(fb, axis=0)
        return fb

    @staticmethod
    def spec2erb(spec: np.ndarray, erb_fb: np.ndarray, db: bool = True):
        """
        ERB filterbank and transform to decibel scale.

        :param spec: Spectrum of shape [B, C, T, F].
        :param erb_fb: ERB filterbank array of shape [B] containing the ERB widths,
                where B are the number of ERB bins.
        :param db: Whether to transform the output into decibel scale. Defaults to `True`.
        :return:
        """
        # complex spec to power spec. (real * real + image * image)
        spec_ = np.abs(spec) ** 2

        # spec to erb feature.
        erb_feat = np.matmul(spec_, erb_fb)

        if db:
            erb_feat = 10 * np.log10(erb_feat + 1e-10)

        erb_feat = np.array(erb_feat, dtype=np.float32)
        return erb_feat


class ErbBands(nn.Module):
    def __init__(self,
                 sample_rate: int = 8000,
                 nfft: int = 512,
                 erb_bins: int = 32,
                 min_freq_bins_for_erb: int = 2,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.erb_bins = erb_bins
        self.min_freq_bins_for_erb = min_freq_bins_for_erb

        erb_fb, erb_fb_inv = self.init_erb_fb()
        erb_fb = torch.tensor(erb_fb, dtype=torch.float32, requires_grad=False)
        erb_fb_inv = torch.tensor(erb_fb_inv, dtype=torch.float32, requires_grad=False)
        self.erb_fb = nn.Parameter(erb_fb, requires_grad=False)
        self.erb_fb_inv = nn.Parameter(erb_fb_inv, requires_grad=False)

    def init_erb_fb(self):
        erb_widths = ErbBandsNumpy.get_erb_widths(
            sample_rate=self.sample_rate,
            nfft=self.nfft,
            erb_bins=self.erb_bins,
            min_freq_bins_for_erb=self.min_freq_bins_for_erb,
        )
        erb_fb = ErbBandsNumpy.get_erb_filter_bank(
            erb_widths=erb_widths,
            normalized=True,
            inverse=False,
        )
        erb_fb_inv = ErbBandsNumpy.get_erb_filter_bank(
            erb_widths=erb_widths,
            normalized=True,
            inverse=True,
        )
        return erb_fb, erb_fb_inv

    def erb_scale(self, spec: torch.Tensor, db: bool = True):
        # spec shape: (b, t, f)
        spec_erb = torch.matmul(spec, self.erb_fb)
        if db:
            spec_erb = 10 * torch.log10(spec_erb + 1e-10)
        return spec_erb

    def erb_scale_inv(self, spec_erb: torch.Tensor):
        spec = torch.matmul(spec_erb, self.erb_fb_inv)
        return spec


def main():

    erb_bands = ErbBands()

    spec = torch.randn(size=(2, 199, 257), dtype=torch.float32)
    spec_erb = erb_bands.erb_scale(spec)
    print(spec_erb.shape)

    spec = erb_bands.erb_scale_inv(spec_erb)
    print(spec.shape)

    return


if __name__ == "__main__":
    main()
