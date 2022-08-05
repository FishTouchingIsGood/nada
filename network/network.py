import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, kernel_size=(3, 3)),
            nn.InstanceNorm2d(channel),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, kernel_size=(3, 3)),
            nn.InstanceNorm2d(channel),
            nn.ReLU(),
        )
        self.block = block

    def forward(self, input):
        return self.block(input) + input


class Unet(nn.Module):
    def __init__(self, device):
        super().__init__()

        # 3 w h
        preprocess = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, kernel_size=(9, 9)),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        # 32 w h
        conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        # 64 w/2 h/2
        conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        # 128 w/4 h/4
        conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        # 256 w/8 h/8

        res = nn.Sequential(*[ResBlock(256) for _ in range(4)])
        # 256 w/8 h/8

        upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

        upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )

        upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU()
        )

        deconv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )

        deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU()
        )

        postprocess = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=(9, 9)),
            nn.Tanh()
        )
        # 3 w h

        self.device = device
        self.preprocess = preprocess.to(device)
        self.conv1 = conv1.to(device)
        self.conv2 = conv2.to(device)
        self.conv3 = conv3.to(device)
        self.res = res.to(device)
        self.deconv1 = deconv1.to(device)
        self.deconv2 = deconv2.to(device)
        self.deconv3 = deconv3.to(device)
        self.upsample1 = upsample1.to(device)
        self.upsample2 = upsample2.to(device)
        self.upsample3 = upsample3.to(device)
        self.postprocess = postprocess.to(device)

    def forward(self, input):

        input = input.to(self.device)

        l1 = self.preprocess(input)
        l2 = self.conv1(l1)
        l3 = self.conv2(l2)
        l4 = self.conv3(l3)

        r4 = self.res(l4)
        r3p = self.upsample3(r4)
        r3f = torch.cat([l3, r3p], dim=1)
        r3 = self.deconv3(r3f)
        r2p = self.upsample2(r3)
        r2f = torch.cat([l2, r2p], dim=1)
        r2 = self.deconv2(r2f)
        r1p = self.upsample1(r2)
        r1f = torch.cat([l1, r1p], dim=1)
        r1 = self.deconv1(r1f)

        output = self.postprocess(r1)

        return output


