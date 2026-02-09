#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch.nn as nn


class Highway(nn.Module):
    """
    https://arxiv.org/abs/1505.00387
    [Submitted on 3 May 2015 (v1), last revised 3 Nov 2015 (this version, v2)]

    discuss of Highway and ResNet
    https://www.zhihu.com/question/279426970
    """
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


if __name__ == '__main__':
    pass
