import os
import glob
import logging
import random
import torchvision.transforms as T
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset

# 以下库是yousurf使用的
import torchio as tio
from torchio.transforms import (
    ZNormalization,
    Compose,
)


def get_data_k(data_root, k=0, i=0, is_train=True):
    """
    Store all raws and gts path into return parameters.
    Args:
        k: 数据被分割成k等分，k=1时不分割，直接随机分成训练集和验证集（4:1）
        i: the ith fold
        data_root: the path of raw and gt folder.
        is_train: True for train or validate. False for tests.
    """
    raws = []
    gts = []

    if is_train:
        sub_dir = 'train'
        # 获取所有图像和ground truth的路径
        raw_path = os.path.join(data_root, sub_dir, 'raw')
        gt_path = os.path.join(data_root, sub_dir, 'gt')
        # for file in glob.glob(os.path.join(raw_path, '*.nii.gz')):
        #     raw_name = os.path.basename(file)[:-7]
        #     gt_name = raw_name + '_GT.nii.gz'
        #     raws.append(file)
        #     gts.append(os.path.join(gt_path, gt_name))
        for raw in glob.glob(os.path.join(raw_path, '*.nii.gz')):
            base_name = os.path.basename(raw)[:-7]
            gt_name = base_name + '_GT.nii.gz'
            gt = os.path.join(gt_path, gt_name)
            raws.append(raw)
            gts.append(gt)

        if k == 1:
            raws_train = []
            gts_train = []
            raws_valid = []
            gts_valid = []
            # 直接随机分成训练集和验证集，train:val = 4:1
            dataset_size = len(raws)
            valid_size = dataset_size // 5
            valid_index = random.sample(range(dataset_size), valid_size)
            train_index = [num for num in range(dataset_size) if num not in valid_index]
            for v in valid_index:
                raws_valid.append(raws[v]), gts_valid.append(gts[v])
            for t in train_index:
                raws_train.append(raws[t]), gts_train.append(gts[t])

            return raws_train, gts_train, raws_valid, gts_valid

        else:
            # 返回第 i+1 折 (i = 0:k-1) 交叉验证时所需要的训练和验证数据，raw_train为训练集，raw_valid为验证集
            fold_size = len(raws) // k  # 每份的个数:数据总条数/折数（组数）
            val_start = i * fold_size
            if i != k - 1:
                val_end = (i + 1) * fold_size
                raws_valid, gts_valid = raws[val_start:val_end], gts[val_start:val_end]
                raws_train = raws[0:val_start] + raws[val_end:]
                gts_train = gts[0:val_start] + gts[val_end:]
            else:  # 若是最后一折交叉验证
                raws_valid, gts_valid = raws[val_start:], gts[val_start:]  # 若不能整除，将多的case放在最后一折里
                raws_train = raws[0:val_start]
                gts_train = gts[0:val_start]
            return raws_train, gts_train, raws_valid, gts_valid

    else:
        sub_dir = 'test'
        raws_test = []
        gts_test = []
        raws_path = os.path.join(data_root, sub_dir, 'raw')
        gts_path = os.path.join(data_root, sub_dir, 'gt')
        files = glob.glob(os.path.join(raws_path, '*.nii.gz'))

        for file in files:
            raw_name = os.path.basename(file)[:-7]
            gt_name = raw_name + '_GT.nii.gz'
            raws_test.append(file)
            gts_test.append(os.path.join(gts_path, gt_name))

        return raws_test, gts_test


class VesselDataTransformer(Dataset):
    """
    Vessel dataset transformer
    """

    def __init__(self, raws, gts, config):

        self.patch_size = config['patch_size']
        self.patch_center = config['patch_center']
        self.raws = raws
        self.gts = gts
        self.to_tensor = T.ToTensor()

        # 下面是yousurf的代码
        self.subjects = []

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, index):
        raw_path = self.raws[index]
        gt_path = self.gts[index]
        raw = nib.load(raw_path)
        raw = raw.get_fdata().astype(np.float32)
        gt = nib.load(gt_path)
        gt = gt.get_fdata().astype(np.float32)

        # 裁减 3-D raw and gt
        raw_patch = fixed_patch_crop(raw, self.patch_center, self.patch_size)
        gt_patch = fixed_patch_crop(gt, self.patch_center, self.patch_size)

        # 对batch进行zero-mean处理，mean -> 0, std -> unit std
        raw_patch = intensity_normalization(raw_patch)

        # toTensor -> (C, z, x ,y)
        raw_patch = self.to_tensor(raw_patch).unsqueeze(0)
        gt_patch = self.to_tensor(gt_patch).unsqueeze(0)
        # raw_patch = raw_patch.transpose(1, 2).transpose(2, 3)
        # gt_patch = gt_patch.transpose(1, 2).transpose(2, 3)
        return raw_patch, gt_patch


def fixed_patch_crop(img, patch_center, patch_size):
    '''
    Fixed center of crop 3D-image or 3-D gt
    img: image or gt tensor
    patch_size: patch size (x, y, z)
    '''

    patch_x, patch_y, patch_z = patch_size
    center_x, center_y, center_z = patch_center

    patch = img[center_x - patch_x // 2: center_x + patch_x // 2, center_y - patch_y // 2: center_y + patch_y // 2,
            center_z - patch_z // 2: center_z + patch_z // 2]

    return patch


def intensity_normalization(dataset):
    mean = dataset.mean()
    std = dataset.std()
    return (dataset - mean) / std
