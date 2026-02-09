#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/modelscope/modelscope/blob/master/modelscope/models/audio/ans/conv_stft.py
"""
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(nfft: int, win_size: int, hop_size: int, win_type: str = None, inverse=False):
    if win_type == "None" or win_type is None:
        window = np.ones(win_size)
    else:
        window = get_window(win_type, win_size, fftbins=True)**0.5

    fourier_basis = np.fft.rfft(np.eye(nfft))[:win_size]
    real_kernel = np.real(fourier_basis)
    image_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, image_kernel], 1).T

    if inverse:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    result = (
        torch.from_numpy(kernel.astype(np.float32)),
        torch.from_numpy(window[None, :, None].astype(np.float32))
    )
    return result


class ConvSTFT(nn.Module):

    def __init__(self,
                 nfft: int,
                 win_size: int,
                 hop_size: int,
                 win_type: str = "hamming",
                 power: int = None,
                 requires_grad: bool = False):
        super(ConvSTFT, self).__init__()

        if nfft is None:
            self.nfft = int(2**np.ceil(np.log2(win_size)))
        else:
            self.nfft = nfft

        kernel, _ = init_kernels(self.nfft, win_size, hop_size, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=requires_grad)

        self.win_size = win_size
        self.hop_size = hop_size

        self.stride = hop_size
        self.dim = self.nfft
        self.power = power

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = torch.unsqueeze(waveform, 1)

        matrix = F.conv1d(waveform, self.weight, stride=self.stride)
        dim = self.dim // 2 + 1
        real = matrix[:, :dim, :]
        imag = matrix[:, dim:, :]
        spec = torch.complex(real, imag)
        # spec shape: [b, f, t], torch.complex64

        if self.power is None:
            return spec
        elif self.power == 1:
            mags = torch.sqrt(real**2 + imag**2)
            # phase = torch.atan2(imag, real)
            return mags
        elif self.power == 2:
            power = real**2 + imag**2
            return power
        else:
            raise AssertionError


class ConviSTFT(nn.Module):

    def __init__(self,
                 win_size: int,
                 hop_size: int,
                 nfft: int = None,
                 win_type: str = "hamming",
                 requires_grad: bool = False):
        super(ConviSTFT, self).__init__()
        if nfft is None:
            self.nfft = int(2**np.ceil(np.log2(win_size)))
        else:
            self.nfft = nfft

        kernel, window = init_kernels(self.nfft, win_size, hop_size, win_type, inverse=True)
        self.weight = nn.Parameter(kernel, requires_grad=requires_grad)
        # weight shape: [f*2, 1, nfft]
        # f = nfft // 2 + 1

        self.win_size = win_size
        self.hop_size = hop_size
        self.win_type = win_type

        self.stride = hop_size
        self.dim = self.nfft

        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_size)[:, None, :])
        # window shape: [1, nfft, 1]
        # enframe shape: [nfft, 1, nfft]

    def forward(self,
                spec: torch.Tensor):
        """
        self.weight shape: [f*2, 1, win_size]
        self.window shape: [1, win_size, 1]
        self.enframe shape: [win_size, 1, win_size]

        :param spec: torch.Tensor, shape: [b, f, t, 2]
        :return:
        """
        spec = torch.view_as_real(spec)
        # spec shape: [b, f, t, 2]
        matrix = torch.concat(tensors=[spec[..., 0], spec[..., 1]], dim=1)
        # matrix shape: [b, f*2, t]

        waveform = F.conv_transpose1d(matrix, self.weight, stride=self.stride)
        # waveform shape: [b, 1, num_samples]

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, matrix.size(-1))**2
        # t shape: [1, win_size, t]
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        # coff shape: [1, 1, num_samples]
        waveform = waveform / (coff + 1e-8)
        # waveform = waveform / coff
        return waveform

    @torch.no_grad()
    def forward_chunk(self,
                      spec: torch.Tensor,
                      cache_dict: dict = None
                      ):
        """
        :param spec: shape: [b, f, t]
        :param cache_dict: dict,
        waveform_cache shape: [b, 1, win_size - hop_size]
        coff_cache shape: [b, 1, win_size - hop_size]
        :return:
        """
        if cache_dict is None:
            cache_dict = defaultdict(lambda: None)
        waveform_cache = cache_dict["waveform_cache"]
        coff_cache = cache_dict["coff_cache"]

        spec = torch.view_as_real(spec)
        matrix = torch.concat(tensors=[spec[..., 0], spec[..., 1]], dim=1)

        waveform_current = F.conv_transpose1d(matrix, self.weight, stride=self.stride)

        t = self.window.repeat(1, 1, matrix.size(-1))**2
        coff_current = F.conv_transpose1d(t, self.enframe, stride=self.stride)

        overlap_size = self.win_size - self.hop_size

        if waveform_cache is not None:
            waveform_current[:, :, :overlap_size] += waveform_cache
        waveform_output = waveform_current[:, :, :self.hop_size]
        new_waveform_cache = waveform_current[:, :, self.hop_size:]

        if coff_cache is not None:
            coff_current[:, :, :overlap_size] += coff_cache
        coff_output = coff_current[:, :, :self.hop_size]
        new_coff_cache = coff_current[:, :, self.hop_size:]

        waveform_output = waveform_output / (coff_output + 1e-8)

        new_cache_dict = {
            "waveform_cache": new_waveform_cache,
            "coff_cache": new_coff_cache,
        }
        return waveform_output, new_cache_dict


def main():
    nfft = 512
    win_size = 512
    hop_size = 256

    stft = ConvSTFT(nfft=nfft, win_size=win_size, hop_size=hop_size, power=None)
    istft = ConviSTFT(nfft=nfft, win_size=win_size, hop_size=hop_size)

    mixture = torch.rand(size=(1, 16000), dtype=torch.float32)
    b, num_samples = mixture.shape
    t = (num_samples - win_size) / hop_size + 1

    spec = stft.forward(mixture)
    b, f, t = spec.shape

    # 如果 spec 是由 stft 变换得来的，以下两种 waveform 还原方法就是一致的，否则还原出的 waveform 会有差异。
    # spec = spec + 0.01 * torch.randn(size=(1, nfft//2+1, t), dtype=torch.float32)
    print(f"spec.shape: {spec.shape}, spec.dtype: {spec.dtype}")

    waveform = istft.forward(spec)
    # shape: [batch_size, channels, num_samples]
    print(f"waveform.shape: {waveform.shape}, waveform.dtype: {waveform.dtype}")
    print(waveform[:, :, 300: 302])

    waveform = torch.zeros(size=(b, 1, num_samples), dtype=torch.float32)
    for i in range(int(t)):
        begin = i * hop_size
        end = begin + win_size
        sub_spec = spec[:, :, i:i+1]
        sub_waveform = istft.forward(sub_spec)
        # (b, 1, win_size)
        waveform[:, :, begin:end] = sub_waveform
    print(f"waveform.shape: {waveform.shape}, waveform.dtype: {waveform.dtype}")
    print(waveform[:, :, 300: 302])

    return


def main2():
    nfft = 512
    win_size = 512
    hop_size = 256

    stft = ConvSTFT(nfft=nfft, win_size=win_size, hop_size=hop_size, power=None)
    istft = ConviSTFT(nfft=nfft, win_size=win_size, hop_size=hop_size)

    mixture = torch.rand(size=(1, 16128), dtype=torch.float32)
    b, num_samples = mixture.shape

    spec = stft.forward(mixture)
    b, f, t = spec.shape

    # 如果 spec 是由 stft 变换得来的，以下两种 waveform 还原方法就是一致的，否则还原出的 waveform 会有差异。
    spec = spec + 0.01 * torch.randn(size=(1, nfft//2+1, t), dtype=torch.float32)
    print(f"spec.shape: {spec.shape}, spec.dtype: {spec.dtype}")

    waveform = istft.forward(spec)
    # shape: [batch_size, channels, num_samples]
    print(f"waveform.shape: {waveform.shape}, waveform.dtype: {waveform.dtype}")
    print(waveform[:, :, 300: 302])

    cache_dict = None
    waveform = torch.zeros(size=(b, 1, num_samples), dtype=torch.float32)
    for i in range(int(t)):
        sub_spec = spec[:, :, i:i+1]
        begin = i * hop_size

        end = begin + win_size - hop_size
        sub_waveform, cache_dict = istft.forward_chunk(sub_spec, cache_dict=cache_dict)
        # end = begin + win_size
        # sub_waveform = istft.forward(sub_spec)

        waveform[:, :, begin:end] = sub_waveform
    print(f"waveform.shape: {waveform.shape}, waveform.dtype: {waveform.dtype}")
    print(waveform[:, :, 300: 302])

    return


if __name__ == "__main__":
    main2()
