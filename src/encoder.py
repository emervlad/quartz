from typing import List

import torch
from torch import nn


class QuartzNetBlock(torch.nn.Module):
    def __init__(
            self,
            feat_in: int,
            filters: int,
            repeat: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            residual: bool,
            separable: bool,
            dropout: float,
    ):

        super().__init__()
        self.residual = residual
        self.separable = separable
        padding_val = self.__class__.get_same_padding(kernel_size, stride, dilation)
        if residual:
            self.res = nn.Sequential(
                nn.Conv1d(in_channels=feat_in, out_channels=filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(num_features=filters)
            )

        if separable:
            layers = []
            layers.append(nn.Conv1d(in_channels=feat_in, out_channels=feat_in,
                                    kernel_size=kernel_size, dilation=dilation, stride=stride, groups=feat_in, padding=padding_val))
            layers.append(nn.Conv1d(in_channels=feat_in, out_channels=filters, kernel_size=1, padding=0))
            layers.append(nn.BatchNorm1d(filters))
            for r in range(1, repeat - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Conv1d(in_channels=filters, out_channels=filters,
                                        kernel_size=kernel_size, dilation=dilation, stride=stride, groups=filters, padding=padding_val))
                layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0))
                layers.append(nn.BatchNorm1d(filters))
            if repeat > 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Conv1d(in_channels=filters, out_channels=filters,
                                        kernel_size=kernel_size, dilation=dilation, stride=stride, groups=filters, padding=padding_val))
                layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0))
                layers.append(nn.BatchNorm1d(filters))
            self.conv = nn.Sequential(*layers)
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channels=feat_in, out_channels=filters,
                                                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_val),
                                      nn.BatchNorm1d(filters))
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

    def get_same_padding(kernel_size, stride, dilation) -> int:
        if stride > 1 and dilation > 1:
            raise ValueError("Only stride OR dilation may be greater than 1")
        return (dilation * (kernel_size - 1)) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        if self.residual:
            b = self.res(x)
        if self.residual:
            x = self.out(a + b)
        else:
            x = self.out(a)
        return x


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
