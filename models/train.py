import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import yaml
from datetime import datetime

from utils import (load_config_ide, load_config_command, get_logger, save_best_checkpoint, save_more_checkpoint,
                   RunningAverageSTD, get_feature_map)
from optimizers import get_optimizer, get_lr_scheduler
from losses import get_loss_criterion
from metrics import get_evaluation_metric
from dataloader import get_data_k, VesselDataTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchinfo import summary
from alive_progress import alive_bar
from torchvision.models.feature_extraction import get_graph_node_names

from models.UNet3D import UNet3D
from models.ERNet import ERNet
from models.RENet import RENet
from models.CS2Net import CS2Net
from models.VNet import VNet
from models.UNet2Plus import UNet2Plus
from models.Uception import Uception


## 下面是我们自己的网络
from models.CONANet import CONANet
from models.CONANet_Base import CONANet_Base
from models.CONANet_WOCL import CONANet_WOCL
from models.CONANet_WOEDGE import CONANet_WOEDGE
from models.CONANet_WOFC import CONANet_WOFC
from models.CONANet_WOLOSS import CONANet_WOLOSS
from models.CONANet_WOCONAM import CONANet_WOCONAM
from models.CONANet_WOCONAM_FC import CONANet_WOCONAM_FC


def main():
    # Load configuration
    config = load_config_ide('configs/IXI_CONANet_config.yml')
    tb_writer = get_logger(config=config, log_level=logging.INFO,
                           log_dir='./logs', only_file=False)
    # Logging configuration
    logging.info('Configuration')
    logging.info(('=' * 65))
    logging.info(yaml.dump(config))

    # Load model (network)
    model_name = config['model'].pop('name')
    model = eval(model_name + "(**config['model'])")
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logging.info(f'**Num of GPU: {torch.cuda.device_count()}\n' + '=' * 65 + '\n')
        model = model.cuda()
    elif torch.cuda.is_available() and config['device'] == 'cpu':
        model = model.cuda()
        logging.info(f'**Num of GPU: 1\n' + '=' * 65 + '\n')
    elif config['device'] == 'cpu':
        assert "DO NOT SUPPORT USING CPU IN THIS PROGRAMME !!!"

    # Load loss function (loss criterion)
    loss_criterion = get_loss_criterion(config['loss'])

    # Load evaluation metrics
    eval_metrics = get_evaluation_metric(config['eval_metric'])

    # Load optimizer
    optimizer = get_optimizer(config['optimizer'], model, loss_criterion)
    # optimizer = get_optimizer(config['optimizer'], model)

    # Load learning rate scheme
    lr_scheduler = get_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    # Logging model and loss architecture
    logging.info('Model Parameters')
    logging.info(summary(model=model))
    logging.info('Loss Parameters')
    logging.info(summary(model=loss_criterion))

    # Load some training parameters
    num_fold = config['train']['num_fold']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    data_path = config['train']['data_loader']['data_path']
    num_workers = config['train']['data_loader']['num_workers']
    best_val_dices = [0. for _ in range(num_fold)]

    # [K-FOLD] START
    t_start = datetime.now()
    for k in range(num_fold):
        logging.info('--' * 20 + ' <The {0:d} fold> '.format(k + 1) + '--' * 20)

        # Load TRAIN and VALIDATION data as k-fold
        raw_train, gt_train, raw_valid, gt_valid = get_data_k(data_path, num_fold, k, is_train=True)
        train_data = VesselDataTransformer(raw_train, gt_train, config['train']['data_loader'])
        validate_data = VesselDataTransformer(raw_valid, gt_valid, config['train']['data_loader'])
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validate_loader = DataLoader(dataset=validate_data, batch_size=1, shuffle=True, num_workers=num_workers)

        for epoch in range(epochs):
            # TRAIN
            trained_model, train_metrics = train(epoch_cnt=epoch + 1,
                                                 model=model,
                                                 train_loader=train_loader,
                                                 loss_criterion=loss_criterion,
                                                 eval_metrics=eval_metrics,
                                                 optimizer=optimizer,
                                                 config=config['train'],
                                                 tb_writer=tb_writer)

            # VALIDATION
            if (epoch + 1) % config['train']['validate_after_epochs'] == 0:
                validate_loss, validate_metrics = validate(validate_model=trained_model,
                                                           validate_loader=validate_loader,
                                                           loss_criterion=loss_criterion,
                                                           eval_metrics=eval_metrics)

                val_dice = validate_metrics[0]
                # According to val_results, dynamically tune lr, default: ReduceLROnPlateau
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(val_dice)
                elif lr_scheduler is not None:
                    lr_scheduler.step(val_dice)
                else:
                    pass

                # Save the trained model with the best validation results
                if val_dice > best_val_dices[k]:
                    best_val_dices[k] = val_dice
                    save_best_checkpoint(state={
                        'epoch_cnt': epoch + 1,
                        'iter_cnt': iter_cnt,
                        'val_dice': val_dice,
                        'model_state_dict': trained_model.state_dict(),
                        'loss_criterion': loss_criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, is_best=True, checkpoint_dir=config['checkpoint_dir'], model_name=model_name, kfold=k+1)
                    logging.info('Saving best model in {0}_fold, DSC: {1:.4f}%'.format(k+1, val_dice * 100))
                else:
                    save_best_checkpoint(state={
                        'epoch_cnt': epoch + 1,
                        'iter_cnt': iter_cnt,
                        'val_dice': val_dice,
                        'model_state_dict': trained_model.state_dict(),
                        'loss_criterion': loss_criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, is_best=False, checkpoint_dir=config['checkpoint_dir'], model_name=model_name, kfold=k+1)

                ## Save the trained model with the validation dice >= 0.9
                # if val_dice >= 0.9:
                #     if val_dice > best_val_dices[k]:
                #         best_val_dices[k] = val_dice
                #     save_more_checkpoint(state={
                #         'epoch_cnt': epoch + 1,
                #         'iter_cnt': iter_cnt,
                #         'val_dice': val_dice,
                #         'model_state_dict': trained_model.state_dict(),
                #         'loss_criterion': loss_criterion.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #     }, checkpoint_dir=config['checkpoint_dir'], model_name=model_name, kfold=k + 1, epoch=epoch + 1)
                #     logging.info(
                #         'Saving model in {0}_fold-{1}_epoch, DSC: {2:.4f}%'.format(k + 1, epoch + 1, val_dice * 100))

    logging.info('K-FOLD VALIDATION AVERAGE DSC: {0:.4f}'.format(np.mean(best_val_dices * 100)))

    # [K-FOLD] END
    t_end = datetime.now()
    logging.info('Finished Training! Elapsed Time: {}'.format(t_end - t_start))


def train(epoch_cnt, model, train_loader, loss_criterion, eval_metrics, optimizer, config, tb_writer):
    """
    Args:
        epoch_cnt: main循环中的第n个epoch
        model: 需要训练的模型
        train_loader: 训练数据，DataLoader
        loss_criterion: 损失函数
        eval_metrics: 评估指标
        optimizer: 优化器，默认使用Adam
        config: Train部分的配置
        tb_writer: tensorboard记录

    Returns:
        model: 训练一个epoch后的模型
        train_losses.avg: 训练平均损失
        train_dice.avg: 训练平均Dice
    """

    # 初始化记录容器
    is_feature = True  # 每个epoch，记录第一个raw数据所有层的feature_maps
    train_losses = RunningAverageSTD()
    train_dice = RunningAverageSTD()
    global iter_cnt

    # 设置为训练模式
    model.train()
    with alive_bar(len(train_loader), bar='classic', spinner='classic', force_tty=True) as bar:
        for index, (raw, gt) in enumerate(train_loader):
            # 前传
            raw, gt = raw.cuda(), gt.cuda()
            pred = model(raw)
            # 计算并记录loss，这里只做了二分类
            # train_loss = loss_criterion(pred, gt)
            train_loss = loss_criterion(pred, gt)  # Only for L_AC
            train_losses.update(train_loss.item(), config['batch_size'])

            # 后传
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 评估，train时最后一位输入为True，只返回dice
            dice = eval_metrics(pred, gt, is_train=True)
            train_dice.update(dice, config['batch_size'])

            # 中间特征图可视化isinstance(model, nn.DataParallel)
            # if is_feature:
            #     feature_layer_names = ['encoder']

            # tensor board plot
            tb_writer.add_scalars('Train', {'Loss': train_loss.item(), 'DSC': dice}, iter_cnt)
            iter_cnt += 1
            bar()

    # 打印每个epoch的结果
    logging.info('[TRAIN: {0:d}|{1:d}] \u2501\u2501\u2501 LOSS:{2:.2f}\tDSC:{3:.2f}%'
                 .format(epoch_cnt, iter_cnt, train_losses.avg, train_dice.avg * 100))

    return model, (train_losses.avg, train_dice.avg)


def validate(validate_model, validate_loader, loss_criterion, eval_metrics):
    # 初始化记录容器
    validate_losses = RunningAverageSTD()
    validate_dice = RunningAverageSTD()
    validate_cldice = RunningAverageSTD()
    validate_hd95 = RunningAverageSTD()
    validate_asd = RunningAverageSTD()
    validate_sen = RunningAverageSTD()
    validate_spec = RunningAverageSTD()

    # 设置为验证模式
    validate_model.eval()
    with torch.no_grad():
        for index, (raw, gt) in enumerate(validate_loader):
            raw, gt = raw.cuda(), gt.cuda()

            # 前传
            pred = validate_model(raw)

            # 计算validate_loss
            # validate_loss = loss_criterion(pred, gt)
            # validate_losses.update(validate_loss.item(), batch_size=1)

            # 评估，validation时最后一位输入为False，返回所有评估指标
            dice, cldice, hd95, asd, sen, spec = eval_metrics(pred, gt, is_train=False)
            validate_dice.update(dice, batch_size=1)
            validate_cldice.update(cldice, batch_size=1)
            validate_hd95.update(hd95, batch_size=1)
            validate_asd.update(asd, batch_size=1)
            validate_sen.update(sen, batch_size=1)
            validate_spec.update(spec, batch_size=1)

        logging.info(
            '[VALIDATE] \u2501\u2501\u2501 '
            # 'LOSS:{0:.2f}\t'
            'DSC:{1:.2f}\u00b1{2:.2f}%\t'
            'clDice:{3:.2f}\u00b1{4:.2f}%\t'
            'HD95:{5:.2f}\u00b1{6:.2f}\t'
            'ASD:{7:.4f}\u00b1{9:.4f}\t'
            'SEN:{9:.2f}\u00b1{10:.2f}\t'
            'SPEC:{11:.2f}\u00b1{12:.2f}\t'.format(0,  # validate_losses.avg,
                                                   validate_dice.avg * 100, validate_dice.std * 100,
                                                   validate_cldice.avg * 100, validate_cldice.std * 100,
                                                   validate_hd95.avg, validate_hd95.std,
                                                   validate_asd.avg, validate_asd.std,
                                                   validate_sen.avg * 100, validate_sen.std * 100,
                                                   validate_spec.avg * 100, validate_spec.std * 100))

        return validate_losses.avg, (validate_dice.avg, validate_hd95.avg, validate_asd.avg)


if __name__ == "__main__":
    # 记录总的iteration次数
    iter_cnt = 0
    main()
