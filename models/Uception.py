# -*- coding: utf-8 -*-
# @Time    : 29/01/2024 02:39
# @Author  : Aaron Yu
# @File    : Uception.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


class inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inception, self).__init__()
        self.branch0 = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),

            nn.Dropout(p=0.25),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 5), padding=(0, 0, 2), bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 5, 1), padding=(0, 2, 0), bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
        )
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Dropout(p=0.25),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),

        )

        self.branch2 = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),

        )

    def forward(self, x):
        x0 = self.branch0(x)
        # print(x0.size())

        x1 = self.branch1(x)
        # print(x1.size())
        x2 = self.branch2(x)
        # print(x2.size())
        out = torch.cat((x0, x1, x2), 1)
        return out


class incep_deep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(incep_deep, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Dropout(p=0.25),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0), bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),

        )
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),

            nn.Dropout(p=0.25),
        )
        self.branch2 = nn.Sequential(

            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), bias=False),
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0), bias=False),

        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),

        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Dropout(p=0.25),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)

        )

    def forward(self, x):
        x0 = self.branch0(x)

        xx = self.branch1(x)
        x2 = self.branch2(xx)
        x3 = self.branch3(xx)
        x4 = self.branch4(xx)
        x5 = self.branch5(x)

        out = torch.cat((x0, x5, x2, x3, x4), 1)
        return out


class Uception(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Uception,self).__init__()

        self.firstconv = nn.Conv3d(input_channels, 64, kernel_size=5, stride=1, padding=2,
                                   bias=False)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstbn = nn.BatchNorm3d(64)
        self.secondconv = nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=2,
                                    bias=False)
        self.dropout = nn.Dropout(p=0.25)
        self.secondrelu = nn.ReLU(inplace=True)
        self.inception1 = inception(64, 20)
        self.bn1 = nn.BatchNorm3d(60)
        self.inception2 = inception(60, 40)
        self.bn2 = nn.BatchNorm3d(120)

        self.inception3 = inception(120, 80)
        self.decoder1 = incep_deep(240, 160)
        self.bn3 = nn.BatchNorm3d(800)
        self.up1 = deconv(800, 160)
        self.decoder2 = incep_deep(280, 80)
        self.bn4 = nn.BatchNorm3d(400)
        self.up2 = deconv(400, 80)
        self.decoder3 = incep_deep(140, 40)
        self.bn5 = nn.BatchNorm3d(200)
        self.up3 = deconv(200, 40)
        self.decoder4 = incep_deep(104, 20)
        self.thirdconv = nn.Conv3d(100, 64, kernel_size=5, stride=1, padding=2,
                                   bias=False)
        self.thirdbn = nn.BatchNorm3d(64)
        self.thirdrelu = nn.ReLU(inplace=True)
        self.finalconv = nn.Conv3d(64, output_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.firstconv(x)
        b1 = self.firstbn(x1)

        r1 = self.firstrelu(b1)
        x2 = self.secondconv(r1)
        d1 = self.dropout(x2)
        r2 = self.secondrelu(d1)

        x2 = self.inception1(r2)
        b2 = self.bn1(x2)
        d2 = self.dropout(b2)

        x3 = self.inception2(d2)
        b3 = self.bn2(x3)
        d3 = self.dropout(b3)

        x4 = self.inception3(d3)
        d4 = self.dropout(x4)

        dec1 = self.decoder1(d4)
        b4 = self.bn3(dec1)
        d5 = self.dropout(b4)
        up1 = self.up1(d5)
        cat1 = torch.cat((d3, up1), 1)

        dec2 = self.decoder2(cat1)
        b5 = self.bn4(dec2)
        d6 = self.dropout(b5)
        up2 = self.up2(d6)
        cat2 = torch.cat((d2, up2), 1)

        dec3 = self.decoder3(cat2)
        b6 = self.bn5(dec3)
        d7 = self.dropout(b6)
        up3 = self.up3(d7)
        cat3 = torch.cat((r2, up3), 1)

        dec4 = self.decoder4(cat3)
        d8 = self.dropout(dec4)
        x5 = self.thirdconv(d8)
        b7 = self.thirdbn(x5)
        r3 = self.thirdrelu(b7)
        d9 = self.dropout(r3)
        x6 = self.finalconv(d9)

        return F.sigmoid(x6)
