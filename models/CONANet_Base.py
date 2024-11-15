# -*- coding: utf-8 -*-
# @Time    : 29/01/2024 04:36
# @Author  : Aaron Yu
# @File    : CONANet_Base.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from functools import partial
from init_weights import init_weights


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ResDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ResDecoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


# class FeatureCombination(nn.Module):
#     def __init__(self, features, branch=2, r=4, L=32):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             branch: the number of branchs.
#             r: the radio for compute d, the length of z.
#             stride: stride, default=1.
#             L: the minimum dim of the vector z, default 32.
#         """
#         super(FeatureCombination, self).__init__()
#         d = max(int(features / r), L)  #
#         self.branch = branch
#         self.features = features
#         self.fc = nn.Linear(features, d)
#         self.softmax = nn.Softmax(dim=1)
#         self.fcs = nn.ModuleList([])
#         for i in range(branch):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
#
#     def forward(self, fea_map1, fea_map2):
#         attention_vectors = torch.Tensor([])
#         fea_maps = torch.cat((fea_map1.unsqueeze_(dim=1), fea_map2.unsqueeze_(dim=1)), dim=1)
#         fea_map1.squeeze_(dim=1)
#         fea_map2.squeeze_(dim=1)
#         fea_U = torch.sum(fea_maps, dim=1)
#         fea_s = fea_U.mean(-1).mean(-1).mean((-1))
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             attention_vectors = torch.cat((attention_vectors, vector), dim=1)
#
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         fea_combine = (fea_maps * attention_vectors).sum(dim=1)
#         return fea_combine


class FeaComDecoder(nn.Module):
    def __init__(self, out_channels):
        super(FeaComDecoder, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1)
        self.ResDecoder = ResDecoder(out_channels)

    def forward(self, x1, x2):
        # add
        # mix = x1 + x2
        # cat
        mix = torch.cat((x1, x2), dim=1)
        mix = self.conv1x1(mix)

        out = self.ResDecoder(mix)
        return out


# class ExtractCenterLine(nn.Module):
#     def __init__(self):
#         super(ExtractCenterLine, self).__init__()
#
#     def soft_erode(self, img):
#         if len(img.shape) == 4:
#             p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
#             p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
#             return torch.min(p1, p2)
#         elif len(img.shape) == 5:
#             p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
#             p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
#             p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
#             return torch.min(torch.min(p1, p2), p3)
#
#     def soft_dilate(self, img):
#         if len(img.shape) == 4:
#             return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
#         elif len(img.shape) == 5:
#             return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
#
#     def forward(self, img, iter_=3):
#         img1 = self.soft_dilate(self.soft_erode(img))
#         skel = F.relu(img - img1)
#         for j in range(iter_):
#             img = self.soft_erode(img)
#             img1 = self.soft_dilate(self.soft_erode(img))
#             delta = F.relu(img - img1)
#             skel = skel + F.relu(delta - skel * delta)
#         return skel


class CONANet_Base(nn.Module):
    def __init__(self, input_channels, output_channels):
        # def __init__(self):
        super(CONANet_Base, self).__init__()

        # ResEncoder
        self.encoder1 = ResEncoder(input_channels, out_channels=32)
        self.encoder2 = ResEncoder(in_channels=32, out_channels=64)
        self.encoder3 = ResEncoder(in_channels=64, out_channels=128)
        self.bridge = ResEncoder(in_channels=128, out_channels=256)
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # ResDecoder
        self.up3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder3 = FeaComDecoder(128)
        self.decoder2 = FeaComDecoder(64)
        self.decoder1 = FeaComDecoder(32)

        self.final = nn.Conv3d(in_channels=32, out_channels=output_channels, kernel_size=1, padding=0)

        # init_weights(self, init_type='normal')
        # for m in self.modules():
        #     init_weights(m, init_type='normal')

    def forward(self, x):

        # Encoder 1
        enc1 = self.encoder1(x)  # @32
        down1 = self.down(enc1)

        # Encoder 2
        enc2 = self.encoder2(down1)  # @64
        down2 = self.down(enc2)

        # Encoder 3
        enc3 = self.encoder3(down2)  # @128
        down3 = self.down(enc3)

        # Bridge
        bridge = self.bridge(down3)  # @256

        # Decoder 3
        up3 = self.up3(bridge)  # @128
        dec3 = self.decoder3(up3, enc3)

        # Decoder 2
        up2 = self.up2(dec3)
        dec2 = self.decoder2(up2, enc2)  # @64

        # Decoder 1
        up1 = self.up1(dec2)
        dec1 = self.decoder1(up1, enc1)  # @32

        # Final
        final = F.sigmoid(self.final(dec1))
        return final
