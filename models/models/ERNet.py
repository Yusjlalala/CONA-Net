# -*- coding: utf-8 -*-
# @Time    : 29/01/2024 02:45
# @Author  : Aaron Yu
# @File    : ERNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.init_weights import init_weights


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


class FSCon(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default=1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(FSCon, self).__init__()
        d = max(int(features / r), L)  #
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)  # original
        self.softmax = nn.Softmax(dim=1)  # original
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, f1, f2):
        feas = torch.cat((f1.unsqueeze_(dim=1), f2.unsqueeze_(dim=1)), dim=1)
        f1.squeeze_(dim=1)
        f2.squeeze_(dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc(fea_s)
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


class FS_Decoder(nn.Module):
    def __init__(self, out_channels):
        super(FS_Decoder, self).__init__()
        self.conv1 = FSCon(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.ResDecoder = ResDecoder(out_channels)

    def forward(self, x1, x2):
        out = self.relu(self.bn1(self.conv1(x1, x2)))
        out = self.ResDecoder(out)
        return out


class ERNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        # def __init__(self):

        super(ERNet, self).__init__()
        self.encoder1 = ResEncoder(input_channels, out_channels=32)
        self.encoder2 = ResEncoder(in_channels=32, out_channels=64)
        self.encoder3 = ResEncoder(in_channels=64, out_channels=128)
        self.bridge = ResEncoder(in_channels=128, out_channels=256)

        # REAM
        self.conv1_1 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(in_channels=256, out_channels=1, kernel_size=1)
        self.convTrans1 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # FSM & Decoder
        self.up3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder3 = FS_Decoder(128)
        self.decoder2 = FS_Decoder(64)
        self.decoder1 = FS_Decoder(32)

        self.final = nn.Conv3d(in_channels=32, out_channels=output_channels, kernel_size=1, padding=0)

        for m in self.modules():
            init_weights(m, init_type='normal')

    def forward(self, x):
        # Encoder 1
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)
        # Encoder 2
        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)
        # REAM 1
        con1_1 = self.conv1_1(enc2)
        convTrans1 = self.convTrans1(con1_1)
        x1 = -1 * (torch.sigmoid(convTrans1)) + 1
        x1 = x1.expand(-1, 32, -1, -1, -1).mul(enc1)
        x1 = x1 + enc1
        # Encoder 3
        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)
        # REAM 2
        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2
        # Bridge
        bridge = self.bridge(down3)
        # REAM 3
        conv3_3 = self.conv3_3(bridge)
        convTrans3 = self.convTrans3(conv3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 128, -1, -1, -1).mul(enc3)
        x3 = x3 + enc3
        # Decoder 3
        up3 = self.up3(bridge)
        dec3 = self.decoder3(up3, x3)
        # Decoder 2
        up2 = self.up2(dec3)
        dec2 = self.decoder2(up2, x2)
        # Decoder 1
        up1 = self.up1(dec2)
        dec1 = self.decoder1(up1, x1)
        # Final
        final = F.sigmoid(self.final(dec1))
        return final
