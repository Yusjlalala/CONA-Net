import os
import logging
import yaml
import torch
import torch.nn as nn
import SimpleITK as sitk
import numpy as np

from torch.utils.data import DataLoader
from alive_progress import alive_bar

from utils import load_config_ide, save_best_checkpoint, RunningAverageSTD, get_reference_geometry
from metrics import get_evaluation_metric
from dataloader import get_data_k, VesselDataTransformer

from models.UNet3D import UNet3D
from models.ERNet import ERNet
from models.RENet import RENet
from models.CS2Net import CS2Net
from models.VNet import VNet
from models.UNet2Plus import UNet2Plus
from models.Uception import Uception

## 下面是我们的网络
from models.CONANet import CONANet
from models.CONANet_Base import CONANet_Base
from models.CONANet_WOCL import CONANet_WOCL
from models.CONANet_WOEDGE import CONANet_WOEDGE
from models.CONANet_WOFC import CONANet_WOFC
from models.CONANet_WOLOSS import CONANet_WOLOSS
from models.CONANet_WOCONAM import CONANet_WOCONAM
from models.CONANet_WOCONAM_FC import CONANet_WOCONAM_FC

def test():

    # 只需要修改这两个地方
    config = load_config_ide('configs/IXI_CONANet_Base_config.yml')
    stamp = 'CONANet_Base_1_fold_best_checkpoint.pth'

    # 加载模型参数
    path = '{}/{}'.format(config['checkpoint_dir'], stamp)
    ckpt = torch.load(path)

    model_name = config['model'].pop('name')
    test_model = eval(model_name + "(**config['model'])")
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        test_model = nn.DataParallel(test_model)
        logging.info(f'Using {torch.cuda.device_count()} GPUs for training/prediction')
        test_model = test_model.cuda()
    elif torch.cuda.is_available() and not config['device'] == 'cpu':
        test_model = test_model.cuda()
    elif config['device'] == 'cpu':
        assert "DO NOT SUPPORT USING CPU IN THIS PROGRAMME !!!"
    # 加载最佳训练模型参数
    test_model.load_state_dict(ckpt['model_state_dict'])
    # 加载评估指标
    eval_criterion = get_evaluation_metric(config['eval_metric'])

    # 加载测试数据集
    raw_test, gt_test = get_data_k(config['test']['data_path'], is_train=False)
    test_data = VesselDataTransformer(raw_test, gt_test, config['train']['data_loader'])
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,
                             num_workers=config['test']['num_workers'])
    ref_imag_path = raw_test[0]

    # 初始化记录容器
    test_dice = RunningAverageSTD()
    test_cldice = RunningAverageSTD()
    test_hd95 = RunningAverageSTD()
    test_asd = RunningAverageSTD()
    test_sen = RunningAverageSTD()
    test_spec = RunningAverageSTD()

    with torch.no_grad():
        # 设置为测试（验证）模式
        test_model.eval()
        with alive_bar(len(test_loader), bar='classic', spinner='classic', force_tty=True) as bar:
            for index, (raw, gt) in enumerate(test_loader):
                # 前传
                raw, gt = raw.cuda(), gt.cuda()
                pred = test_model(raw)

                # 评估
                dice, cldice, hd95, asd, sen, spec = eval_criterion(pred, gt,
                                                                    is_train=False)
                test_dice.update(dice, batch_size=1)
                test_cldice.update(cldice, batch_size=1)
                test_hd95.update(hd95, batch_size=1)
                test_asd.update(asd, batch_size=1)
                test_sen.update(sen, batch_size=1)
                test_spec.update(spec, batch_size=1)

                # pred_name 不需要根据数据集的命名作修改
                pred_name = os.path.basename(raw_test[index])[:-7]
                assert not ('/' in pred_name or '.' in pred_name), (
                    "Prediction name contains '/' or '.', please check the "
                    "file name!")

                # 推理完的数据是[Z, X, Y]，保存前的numpy轴的顺序必须得是[Z, Y, X]！！！
                raw = raw.squeeze(0).squeeze(0).transpose(1, 2).detach().cpu().numpy().astype(np.float64)
                gt = gt.squeeze(0).squeeze(0).transpose(1, 2).detach().cpu().numpy().astype(np.float64)
                pred = pred.squeeze(0).squeeze(0).transpose(1, 2).detach().cpu().numpy().astype(np.float64)
                pred = np.where(pred > 0.5, 1., 0)  # 这里设置一个输出pred的阈值

                raw = get_reference_geometry(sitk.GetImageFromArray(raw), ref_path=ref_imag_path)
                pred = get_reference_geometry(sitk.GetImageFromArray(pred), ref_path=ref_imag_path)
                gt = get_reference_geometry(sitk.GetImageFromArray(gt), ref_path=ref_imag_path)

                # sitk.WriteImage(raw, os.path.join(config['pred_dir'], model_name + '_' + pred_name + '_raw.nii.gz'))
                # sitk.WriteImage(pred, os.path.join(config['pred_dir'], model_name + '_' + pred_name + '_pred.nii.gz'))
                # sitk.WriteImage(gt, os.path.join(config['pred_dir'], model_name + '_' + pred_name + '_gt.nii.gz'))
                # 进度条＋1
                bar()

        print(
            '[TEST] \u2501\u2501\u2501 '
            'DSC:{0:.2f}\u00b1{1:.2f}\t'
            'clDice:{2:.2f}\u00b1{3:.2f}\t'
            'HD95:{4:.2f}\u00b1{5:.2f}\t'
            'ASD:{6:.4f}\u00b1{7:.4f}\t'
            'SEN:{8:.2f}\u00b1{9:.2f}\t'
            'SPEC:{10:.2f}\u00b1{11:.2f}\t'.format(test_dice.avg * 100, test_dice.std * 100,
                                                   test_cldice.avg * 100, test_cldice.std * 100,
                                                   test_hd95.avg, test_hd95.std,
                                                   test_asd.avg, test_asd.std,
                                                   test_sen.avg * 100, test_sen.std * 100,
                                                   test_spec.avg * 100, test_spec.std * 100))


if __name__ == "__main__":
    test()
