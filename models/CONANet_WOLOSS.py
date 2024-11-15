# -*- coding: utf-8 -*-
# @Time    : 15/02/2024 17:33
# @Author  : Aaron Yu
# @File    : CONANet_WOLOSS.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from functools import partial
from init_weights import init_weights

warnings.filterwarnings('ignore')


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
        self.conv1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # residual = self.conv1x1(x)
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class FeatureCombination(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default=1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(FeatureCombination, self).__init__()
        d = max(int(features / r), L)  #
        self.M = M
        self.features = features * 2
        self.fc1 = nn.Linear(self.features, d)  # original
        self.softmax = nn.Softmax(dim=1)  # original
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, self.features // 2)
            )

    def forward(self, f1, f2):
        feas = torch.cat((f1.unsqueeze_(dim=1), f2.unsqueeze_(dim=1)), dim=1)
        f1.squeeze_(dim=1)
        f2.squeeze_(dim=1)
        fea_U = torch.cat((f1, f2), dim=1) # 256*D*H*W
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc1(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_fuse = (feas * attention_vectors).sum(dim=1)
        return fea_fuse


class FeaComDecoder(nn.Module):
    def __init__(self, out_channels):
        super(FeaComDecoder, self).__init__()
        self.conv1 = FeatureCombination(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.ResDecoder = ResDecoder(out_channels)

    def forward(self, x1, x2):
        out = self.relu(self.bn1(self.conv1(x1, x2)))
        out = self.ResDecoder(out)
        return out


class ConnectivityAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConnectivityAttention, self).__init__()
        self.mul_channels = in_channels // 2
        self.bdr_conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.bdr_deconv = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

    def forward(self, low_fea, high_fea, low_cl_fea=None):
        # boundary
        bdr = 1 - torch.sigmoid(self.bdr_deconv(self.bdr_conv(high_fea)))
        bdr = bdr.expand(-1, self.mul_channels, -1, -1, -1).mul(low_fea)
        ca = bdr + low_fea
        # centerline
        if low_cl_fea is not None:
            cl = low_cl_fea.mul(low_fea)
            ca = ca + cl

        return ca

class CONANet_WOLOSS(nn.Module):
    def __init__(self, input_channels, output_channels):
        # def __init__(self):
        super(CONANet_WOLOSS, self).__init__()

        # CenterLine 1
        self.cl_bn = nn.BatchNorm3d(input_channels)
        self.cl_pad = nn.ReflectionPad3d(padding=(1, 1, 1, 1, 1, 1))
        # self.cl_thr = nn.Parameter(torch.Tensor([0.08]))
        self.cl_thr = torch.Tensor([0.1])
        self.cl_relu = nn.ReLU()
        self.cl_encoder1 = ResEncoder(input_channels, out_channels=32)
        self.cl_encoder2 = ResEncoder(in_channels=32, out_channels=64)
        self.cl_encoder3 = ResEncoder(in_channels=64, out_channels=128)

        # CenterLine 2
        # self.cl_soft_skel = ExtractCenterLine()

        # Encoder
        self.encoder1 = ResEncoder(input_channels, out_channels=32)
        self.encoder2 = ResEncoder(in_channels=32, out_channels=64)
        self.encoder3 = ResEncoder(in_channels=64, out_channels=128)
        self.bridge = ResEncoder(in_channels=128, out_channels=256)

        # CAM
        self.conatten1 = ConnectivityAttention(in_channels=64)
        self.conatten2 = ConnectivityAttention(in_channels=128)
        self.conatten3 = ConnectivityAttention(in_channels=256)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # FSM & Decoder
        self.up3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder3 = FeaComDecoder(128)
        self.decoder2 = FeaComDecoder(64)
        self.decoder1 = FeaComDecoder(32)

        self.final = nn.Conv3d(in_channels=32, out_channels=output_channels, kernel_size=1, padding=0)

        # init_weights(self, init_type='normal')
        # for m in self.modules():
        #     init_weights(m, init_type='kaiming')

    def forward(self, x):
        # 如果注释CenterLine 1，就去掉下面三个的注释
        # centerline1 = None
        # centerline2 = None
        # centerline3 = None

        ## CenterLine
        cl_kernel = torch.tensor(data=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26,
                                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                 dtype=torch.float32).reshape(1, 1, 3, 3, 3).cuda()
        centerline = F.conv3d(self.cl_pad(x), cl_kernel)
        centerline = self.cl_relu(centerline)
        centerline = (centerline - torch.min(centerline)) / (torch.max(centerline) - torch.min(centerline))
        centerline[centerline < self.cl_thr.cuda()] = 0.  # 滤除背景噪声
        # print("cl_thr: ({0:.4f}, {1})".format(self.cl_thr.item(),
        #                                       self.cl_thr.grad.item() if self.cl_thr.grad.item() is not None else 0))

        # Encoder 1
        centerline1 = self.cl_encoder1(centerline)
        # centerline1 = self.encoder1(centerline)
        centerline_down1 = self.maxpool(centerline1)
        enc1 = self.encoder1(x)
        maxpool1 = self.maxpool(enc1)

        # Encoder 2
        centerline2 = self.cl_encoder2(centerline_down1)
        # centerline2 = self.encoder2(centerline_down1)
        centerline_down2 = self.maxpool(centerline2)
        enc2 = self.encoder2(maxpool1)
        maxpool2 = self.maxpool(enc2)

        # CAM 1
        connectivity_attention1 = self.conatten1(enc1, enc2, centerline1)

        # Encoder 3
        centerline3 = self.cl_encoder3(centerline_down2)
        # centerline3 = self.encoder3(centerline_down2)
        enc3 = self.encoder3(maxpool2)
        maxpool3 = self.maxpool(enc3)

        # CAM 2
        connectivity_attention2 = self.conatten2(enc2, enc3, centerline2)

        # Bridge
        bridge = self.bridge(maxpool3)

        # CAM 3
        connectivity_attention3 = self.conatten3(enc3, bridge, centerline3)

        # Decoder 3
        up3 = self.up3(bridge)
        dec3 = self.decoder3(up3, connectivity_attention3)

        # Decoder 2
        up2 = self.up2(dec3)
        dec2 = self.decoder2(up2, connectivity_attention2)

        # Decoder 1
        up1 = self.up1(dec2)
        dec1 = self.decoder1(up1, connectivity_attention1)

        # Final
        final = F.sigmoid(self.final(dec1))
        return final
