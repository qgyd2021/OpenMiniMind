#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np


class MelBandsNumpy(object):
    @staticmethod
    def freq2mel(freq_hz):
        return 2595.0 * np.log10(1.0 + freq_hz / 700.0)

    @staticmethod
    def mel2freq(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def get_mel_points(cls,
                       n_mels: int,
                       f_min: float,
                       f_max: float,
                       ):
        mel_min = cls.freq2mel(f_min)
        mel_max = cls.freq2mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

        freq_points = cls.mel2freq(mel_points)
        return freq_points

    @classmethod
    def get_mel_filter_bank(cls,
                            n_fft: int,
                            n_mels: int,
                            sample_rate: int,
                            f_min: float = 0.0,
                            f_max: float = None,
                            ):
        if f_max is None:
            f_max = float(sample_rate) / 2.0

        n_freqs = n_fft // 2 + 1

        freq_points = cls.get_mel_points(n_mels=n_mels,
                                         f_min=f_min,
                                         f_max=f_max)

        # freq bin 索引
        fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

        # 构造三角滤波组
        mel_fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
        for m in range(n_mels):
            f_left = freq_points[m]
            f_center = freq_points[m + 1]
            f_right = freq_points[m + 2]

            # 上升带
            left_slope = (fft_freqs - f_left) / (f_center - f_left)
            # 下降带
            right_slope = (f_right - fft_freqs) / (f_right - f_center)

            mel_fb[:, m] = np.maximum(0.0, np.minimum(left_slope, right_slope))

        # 归一化
        mel_fb /= np.maximum(mel_fb.sum(axis=0, keepdims=True), 1e-10)
        return torch.from_numpy(mel_fb)


class MelBands(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        n_mels: int = 64,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max

        # mel_fb shape: [freq_bins, mel_bins]
        mel_fb = MelBandsNumpy.get_mel_filter_bank(
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.register_buffer("mel_fb", mel_fb)

    def mel_scale(self, spec: torch.Tensor) -> torch.Tensor:
        # spec shape: (b, t, freq_bins)
        mel_out = torch.matmul(spec, self.mel_fb)
        # mel_out shape: (b, t, mel_bins)
        return mel_out


def main():
    spec = torch.randn(2, 199, 257)  # (batch, time, freq_bins)
    mel_layer = MelBands(n_fft=512, n_mels=80, sample_rate=16000)
    mel_feat = mel_layer.mel_scale(spec)  # (2, 199, 80)
    print(mel_feat.shape)
    return


if __name__ == "__main__":
    main()
