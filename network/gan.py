import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


class AdaIn(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channel)
        self.channel = channel

    def forward(self, mat, scale, bias):
        scale = scale.reshape(1, self.channel, 1, 1)
        scale = scale.expand(mat.shape)
        bias = bias.reshape(1, self.channel, 1, 1)
        bias = bias.expand(mat.shape)
        return mat * scale + bias


class MyConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=0):
        super().__init__()
        self.block = nn.Sequential(
            # nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, mat):
        return self.block(mat)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), output_padding=(0, 0)):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, mat):
        return self.block(mat)




class Layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.unsample = Upsample(in_channel,
                                 out_channel,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1),
                                 output_padding=(1, 1))
        self.conv1 = MyConv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1)
        self.adain1 = AdaIn(out_channel)
        self.conv2 = MyConv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1)
        self.adain2 = AdaIn(out_channel)
        self.linear1s = nn.Linear(512, out_channel)
        self.linear1b = nn.Linear(512, out_channel)
        self.linear2s = nn.Linear(512, out_channel)
        self.linear2b = nn.Linear(512, out_channel)


    def forward(self, mat, style):
        mat = self.unsample(mat)
        mat = self.conv1(mat)
        s1 = self.linear1s(style)
        b1 = self.linear1b(style)
        mat = self.adain1(mat, s1, b1)
        mat = self.conv2(mat)
        s2 = self.linear2s(style)
        b2 = self.linear2b(style)
        mat = self.adain2(mat, s2, b2)
        return mat

class Preprocess(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.adain1 = AdaIn(channel)
        self.conv = MyConv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.adain2 = AdaIn(channel)
        self.linear1s = nn.Linear(512, channel)
        self.linear1b = nn.Linear(512, channel)
        self.linear2s = nn.Linear(512, channel)
        self.linear2b = nn.Linear(512, channel)

    def forward(self, mat, style):
        s1 = self.linear1s(style)
        b1 = self.linear1b(style)
        mat = self.adain1(mat, s1, b1)
        mat = self.conv(mat)
        s2 = self.linear2s(style)
        b2 = self.linear2b(style)
        mat = self.adain2(mat, s2, b2)
        return mat

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 512 7 7
        self.preprocess = Preprocess(512)
        # 512 7 7
        self.block1 = Layer(512, 256)
        # 256 14 14
        self.block2 = Layer(256, 128)
        # 128 28 28
        self.block3 = Layer(128, 64)
        # 64 56 56
        self.block4 = Layer(64, 32)
        # 32 112 112
        self.block5 = Layer(32, 3)
        # 3 224 224
        self.tan = nn.Tanh()

    def forward(self, mat, style):
        mat = self.preprocess(mat, style)
        mat = self.block1(mat, style)
        mat = self.block2(mat, style)
        mat = self.block3(mat, style)
        mat = self.block4(mat, style)
        mat = self.block5(mat, style)
        mat = self.tan(mat)
        return mat
