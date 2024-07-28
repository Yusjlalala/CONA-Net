import argparse
import logging
import os
import sys
import time
import yaml
import torch
import shutil
import SimpleITK as sitk
import numpy as np
import torch.nn as nn

from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.utils.tensorboard import SummaryWriter


def get_logger(config, log_level=logging.INFO,
               log_dir='./logs', only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = (config['train']['data_loader']['dataset_name'] + '_' + config['model']['name'])

    # 在logs文件夹下创建每次实验对应的子文件夹
    log_sub_dir = os.path.join(log_dir, log_file_name + '_' + time.strftime('%Y%m%d_%H%M%S',
                                                                            time.localtime(time.time())))
    os.makedirs(log_sub_dir)

    log_file = os.path.join(log_sub_dir, 'log_file.txt')
    formatter = '%(message)s'
    datefmt = "%Y-%d-%m %H:%M:%S"
    if only_file:
        logging.basicConfig(filename=log_file, level=log_level,
                            format=formatter, datefmt=datefmt)
    else:
        logging.basicConfig(level=log_level, format=formatter, datefmt=datefmt,
                            handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler(sys.stdout)])

    # 在这里初始化 tensorboard logs
    writer = SummaryWriter(log_dir=log_sub_dir)
    return writer


def load_config_command():
    # 这里通过命令行载入Config，如果用IDE则不需要这部分
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)

    device = config.get('device', None)
    if device == 'cpu':
        logging.warning('CPU mode forced in config, this will likely result in slow training/prediction')
        config['device'] = 'cpu'
        return config

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        logging.warning('CUDA not available, using CPU')
        config['device'] = 'cpu'
    return config


def load_config_ide(config_path):
    # 这里通过IDE载入Config
    config = _load_config_yaml(config_path)
    device = config.get('device', None)
    if device == 'cpu':
        logging.warning('CPU mode forced in config, this will likely result in slow training/prediction')
        config['device'] = 'cpu'
        return config

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        logging.warning('CUDA not available, using CPU')
        config['device'] = 'cpu'
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def save_best_checkpoint(state, is_best, checkpoint_dir, model_name, kfold):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        kfold: the kth-fold of the validation
        model_name: Model Name
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if is_best:
        best_file_path = os.path.join(checkpoint_dir, model_name + '_' + str(kfold) + '_fold_best_checkpoint.pth')
        torch.save(state, best_file_path)
    else:
        last_file_path = os.path.join(checkpoint_dir, model_name + '_' + str(kfold) + '_fold_last_checkpoint.pth')
        torch.save(state, last_file_path)


def save_more_checkpoint(state, checkpoint_dir, model_name, kfold, epoch):
    """Saves model

    Args:
        epoch: 显示第几个epoch
        kfold: the kth-fold of the validation
        model_name: Model Name
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    file_path = os.path.join(checkpoint_dir, model_name + '_' + str(kfold) + 'fold_' + str(epoch) + 'epoch_' + 'checkpoint.pth')
    torch.save(state, file_path)



def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:`
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def get_reference_geometry(image: sitk.Image, ref_path: sitk.Image):
    '''
    读取参考图像的几何参数，以确保输出的正确性
    Args:
        image: 需要对齐的图像
        ref_path: 对齐的参考图像

    Returns:
        image:

    '''
    ref = sitk.ReadImage(ref_path)

    # image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def get_feature_map(model, raw, epoch_cnt, feature_layer_names=None):
    with torch.no_grad():
        # 画原始图像
        raw_image = raw.transpose(3, 4).detach().cpu().numpy()[0, 0, raw.shape[2] // 2, :, :].astype("float32")
        plt.title("raw")
        plt.grid(False)
        plt.axis("off")
        plt.imshow(raw_image)
        plt.show()

        # 并行运算时需要此行
        if isinstance(model, nn.DataParallel):
            model = model.module

        if feature_layer_names is None:
            feature_layer_names = [name[0] for name in model.named_children()]

        for feature_layer_name in feature_layer_names:

            feature_extractor = create_feature_extractor(model, return_nodes=[feature_layer_name])
            feature_map = feature_extractor(raw)
            feature_map = feature_map[feature_layer_name].transpose(3, 4).detach().cpu().numpy()

            n_features = feature_map.shape[1]  # [N, C, Z, Y, X]
            image_size = feature_map.shape[-1]  # 特征图谱的大小
            n_rows = 16
            n_cols = n_features // n_rows

            display_grid = np.zeros(((image_size + 1) * n_cols - 1, n_rows * (image_size + 1) - 1))
            for col in range(n_cols):
                for row in range(n_rows):
                    channel_index = col * n_rows + row
                    channel_image = feature_map[0, channel_index, feature_map.shape[2] // 2, :, :].copy()
                    if channel_image.sum() != 0:  # 数据处理，使其适合于作为图像展示
                        channel_image /= channel_image.max()
                        channel_image *= 255
                    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                    display_grid[col * (image_size + 1): (col + 1) * image_size + col,
                    row * (image_size + 1): (row + 1) * image_size + row] = channel_image

            scale = 1. / image_size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title("Epoch-{}_{}".format(epoch_cnt, feature_layer_name))
            plt.grid(False)
            plt.axis("off")
            plt.imshow(display_grid, aspect="auto", cmap="viridis")
            plt.show()


class RunningAverageSTD:
    """Computes and stores the average
    """

    def __init__(self):
        # 计算均值和标准差
        self.metrics_list = []
        self.avg = 0.
        self.std = 0.

    def update(self, value, batch_size):
        for _ in range(batch_size):
            self.metrics_list.append(value)
        self.avg = np.average(self.metrics_list)
        self.std = np.std(self.metrics_list)
