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
        mat = self.norm(mat)
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
        # noise = torch.randn(mat.shape,device=mat.device)*0.0002
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

    def freeze_linear(self):
        for x in self.linear1s.parameters():
            x.requires_grad = False
        for x in self.linear2s.parameters():
            x.requires_grad = False
        for x in self.linear1b.parameters():
            x.requires_grad = False
        for x in self.linear2b.parameters():
            x.requires_grad = False
        for x in self.conv1.parameters():
            x.requires_grad = True
        for x in self.conv2.parameters():
            x.requires_grad = True

    def freeze_conv(self):
        for x in self.linear1s.parameters():
            x.requires_grad = True
        for x in self.linear2s.parameters():
            x.requires_grad = True
        for x in self.linear1b.parameters():
            x.requires_grad = True
        for x in self.linear2b.parameters():
            x.requires_grad = True
        for x in self.conv1.parameters():
            x.requires_grad = False
        for x in self.conv2.parameters():
            x.requires_grad = False


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

    def freeze_linear(self):
        for x in self.linear1s.parameters():
            x.requires_grad = False
        for x in self.linear2s.parameters():
            x.requires_grad = False
        for x in self.linear1b.parameters():
            x.requires_grad = False
        for x in self.linear2b.parameters():
            x.requires_grad = False
        for x in self.conv.parameters():
            x.requires_grad = True

    def freeze_conv(self):
        for x in self.linear1s.parameters():
            x.requires_grad = True
        for x in self.linear2s.parameters():
            x.requires_grad = True
        for x in self.linear1b.parameters():
            x.requires_grad = True
        for x in self.linear2b.parameters():
            x.requires_grad = True
        for x in self.conv.parameters():
            x.requires_grad = False



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 512 4 4
        self.preprocess = Preprocess(512)
        # 512 4 4
        self.block1 = Layer(512, 512)
        # 512 8 8
        self.block2 = Layer(512, 512)
        # 512 16 16
        self.block3 = Layer(512, 512)
        # 512 32 32
        self.block4 = Layer(512, 256)
        # 256 64 64
        self.block5 = Layer(256, 128)
        # 128 128 128
        self.block6 = Layer(128, 64)
        # 64 256 256
        self.block7 = Layer(64, 16)
        # 32 512 512
        # self.block8 = Layer(32, 3)
        # # 3 1024 1024

        self.final = nn.Sequential(
            # nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            # nn.AdaptiveAvgPool2d(896),
            # nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),
            # nn.AdaptiveAvgPool2d(448),
            # nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),
            # nn.AdaptiveAvgPool2d(224),

            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(2),
            # nn.AdaptiveAvgPool2d(224),
        )

        self.tan = nn.Tanh()


    def forward(self, mat, style):
        mat = self.preprocess(mat, style)
        mat = self.block1(mat, style)
        mat = self.block2(mat, style)
        mat = self.block3(mat, style)
        mat = self.block4(mat, style)
        mat = self.block5(mat, style)
        mat = self.block6(mat, style)
        mat = self.block7(mat, style)
        # mat = self.block8(mat, style)
        # mat = torch.nn.functional.interpolate(mat, (224,224), mode="bilinear")

        mat = self.final(mat)
        mat = self.tan(mat)
        return mat

    def freeze_linear(self):
        self.preprocess.freeze_linear()
        self.block1.freeze_linear()
        self.block2.freeze_linear()
        self.block3.freeze_linear()
        self.block4.freeze_linear()
        self.block5.freeze_linear()
        self.block6.freeze_linear()
        self.block7.freeze_linear()
        # self.block8.freeze_linear()
        for x in self.final.parameters():
            x.requires_grad = True


    def freeze_conv(self):
        self.preprocess.freeze_conv()
        self.block1.freeze_conv()
        self.block2.freeze_conv()
        self.block3.freeze_conv()
        self.block4.freeze_conv()
        self.block5.freeze_conv()
        self.block6.freeze_conv()
        self.block7.freeze_conv()
        # self.block8.freeze_conv()
        for x in self.final.parameters():
            x.requires_grad = False
