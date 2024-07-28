import time
import torch
import torch.nn.functional as F

from torch import nn as nn
from metrics import compute_per_channel_dice, DiceCoefficient
# from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


# 建立各种Loss的类
#######################################################################################################################

class BCELoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        assert pred.view(-1)[torch.argmax(pred)] <= 1, 'Please ensure model final output is nn.Sigmoid()'
        return self.bce(pred, target)


class DiceLoss(nn.Module):
    """
    # Input is expected to be probabilities instead of logits, input和groundtruth都是归一化的tensor

    # 二分类或单一对象分割问题：通常用于binary数据，使用sigmoid进行channel标准化

    # 多类分割问题：使用softmax进行channel标准化
    """

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, pred, groundtruth):
        # 对所有通道的dice求平均得到一次iteration的Dice
        return 1. - torch.mean(compute_per_channel_dice(pred, groundtruth))


class EdgeReinforceLoss(nn.Module):
    """ Loss for ER-Net

    Input is expected to be probabilities instead of logits, and groundtruth should be normalized.

    """

    def __init__(self, threshold=0.8):
        super(EdgeReinforceLoss, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]).cuda().requires_grad_())  # 将权重设置为浮点数
        self.weight2 = nn.Parameter(torch.Tensor([1.]).cuda().requires_grad_())  # 将权重设置为浮点数
        self.dice_coeff = DiceCoefficient()
        self.dice_loss = DiceLoss()
        self.edge_dice_loss = DiceLoss()
        self.edge_bce_loss = nn.BCELoss()
        self.er_threshold = threshold

    def forward(self, pred, groundtruth):
        # 设置拉普拉斯卷积核
        laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3, 3).cuda()

        # 用拉普拉斯卷积核提取gt的边界信息
        boundary_pred = F.conv3d(pred, laplacian_kernel, padding=1)
        boundary_groundtruth = F.conv3d(groundtruth, laplacian_kernel, padding=1)
        # 这个值指只会大于0，被clamp在[min,]
        boundary_pred = boundary_pred.clamp(min=0)
        boundary_pred[boundary_pred > 0.1] = 1
        boundary_pred[boundary_pred <= 0.1] = 0
        boundary_groundtruth = boundary_groundtruth.clamp(min=0)
        boundary_groundtruth[boundary_groundtruth > 0.1] = 1
        boundary_groundtruth[boundary_groundtruth <= 0.1] = 0

        # 用最邻近插值使得边界gt与input对齐
        if pred.shape[-1] != boundary_groundtruth.shape[-1]:
            boundary_groundtruth = F.interpolate(
                boundary_groundtruth, pred.shape[2:], mode='nearest')

        # 根据论文计算DSC作为阈值判断标准
        dice_coeff = self.dice_coeff(pred, groundtruth)

        # 计算loss
        dice_loss = self.dice_loss(pred, groundtruth)
        # edge_dice_loss = self.edge_dice_loss(pred, boundary_groundtruth)
        # edge_bce_loss = self.edge_bce_loss(pred, boundary_groundtruth)
        edge_dice_loss = self.edge_dice_loss(boundary_pred, boundary_groundtruth)
        edge_bce_loss = self.edge_bce_loss(boundary_pred, boundary_groundtruth)

        if dice_coeff.float() < self.er_threshold:
            total_loss = dice_loss
        else:
            total_loss = (dice_loss
                          + self.weight1.pow(-2) * edge_bce_loss
                          + self.weight2.pow(-2) * edge_dice_loss
                          + (1 + self.weight1 * self.weight2).log())

        return total_loss


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, pred, target):
        assert pred.view(-1)[torch.argmax(pred)] <= 1, 'Please ensure model final output is nn.Sigmoid()'
        return self.alpha * self.bce(pred, target) + self.beta * self.dice(pred, target)


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftCLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, groundtruth, pred):
        skel_pred = self.soft_skel(pred, self.iter)
        skel_true = self.soft_skel(groundtruth, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, groundtruth)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, pred)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img, iter_):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for j in range(iter_):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class SoftDiceCCLDiceLoss(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, cl_smooth=1.):
        super(SoftDiceCCLDiceLoss, self).__init__()
        self.iter = iter_
        self.cl_smooth = cl_smooth
        self.alpha = alpha

    def forward(self, groundtruth, pred):
        dice = self.soft_dice(groundtruth, pred)
        skel_pred = self.soft_skel(pred, self.iter)
        skel_true = self.soft_skel(groundtruth, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, groundtruth)[:, 1:, ...]) + self.cl_smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.cl_smooth)
        tsens = (torch.sum(torch.multiply(skel_true, pred)[:, 1:, ...]) + self.cl_smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.cl_smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice

    def soft_dice(self, groundtruth, pred):
        """[function to compute dice loss]

        Args:
            groundtruth ([float32]): [ground truth image]
            pred ([float32]): [predicted image]

        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        intersection = torch.sum((groundtruth * pred)[:, 1:, ...])
        coeff = (2. * intersection + smooth) / (
                torch.sum(groundtruth[:, 1:, ...]) + torch.sum(pred[:, 1:, ...]) + smooth)
        return 1. - coeff

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img, iter_):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for j in range(iter_):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class EdgeDiceCLDiceLoss(nn.Module):
    """ Loss for Ours, (CenterLine + BCEEdge + Dice) Reinforce Loss

    Input is expected to be probabilities instead of logits, and groundtruth should be normalized.

    """

    def __init__(self, threshold=0.8):
        super(EdgeDiceCLDiceLoss, self).__init__()

        self.w1 = nn.Parameter(torch.Tensor([1.]))  # 0.9802
        self.w2 = nn.Parameter(torch.Tensor([1.]))  # 0.4141

        # self.w1 = torch.Tensor([0.9802])
        # self.w2 = torch.Tensor([0.4141])
        self.w3 = torch.Tensor([1.])
        self.alpha = torch.Tensor([0.5])

        self.dice_coeff = DiceCoefficient()
        self.dice_loss = DiceLoss()
        self.dice_edge_loss = DiceLoss()
        self.bce_edge_loss = nn.BCELoss()
        self.threshold = threshold
        self.iter = 3
        self.smooth = 1.

    def forward(self, pred, groundtruth):
        # 计算Dice作为阈值判断标准
        dice_coeff = self.dice_coeff(pred, groundtruth)

        # 计算部分loss
        dice_loss = self.dice_loss(pred, groundtruth)
        dice_edge_loss, bce_edge_loss = self.dice_bce_edge_loss(pred, groundtruth)
        cl_dice_loss = self.cldice_loss(pred, groundtruth)

        # 整合Loss
        if dice_coeff.float() < self.threshold:
            total_loss = dice_loss
        else:
            ## 这里不要动了！！！，当w1,w2不可学习的时候达到Dice最大
            total_loss = dice_loss + (self.w1.pow(-2) * dice_edge_loss
                                      + self.w2.pow(-2) * cl_dice_loss
                                      + (1 + self.w1.abs() * self.w2.abs()).log())

            # total_loss = 0.4 * dice_loss + 0.3 * cl_dice_loss + 0.3 * (self.w1.pow(-2) * dice_edge_loss
            #                                                            + self.w2.pow(-2) * bce_edge_loss
            #                                                            + (1 + self.w1 * self.w2).log())

            # total_loss = ((1 - self.alpha.cuda()) * dice_loss + self.alpha.cuda() *
            #               (self.w1.pow(-2).cuda() * dice_edge_loss + self.w2.pow(-2).cuda() * cl_dice_loss
            #               + (1 + self.w1.cuda() * self.w2).log()))

            # total_loss = dice_loss + (self.w1.pow(-2) * dice_edge_loss
            #                           + self.w2.pow(-2) * bce_edge_loss
            #                           + self.w3.pow(-2) * cl_dice_loss
            #                           + (1 + self.w1 * self.t3).log())

            # total_loss = dice_loss + (#self.w1.pow(-2) * (dice_edge_loss + bce_edge_loss) +
            #                           self.w2.pow(-2) * cl_dice_loss +
            #                           self.w3.pow(-2) * dice_loss +
            #                           (1 + self.w1 * self.w2 * self.w3).log())

        print("(w1, grad1):({0:.4f}, {1:.4f})  (w2, grad2):({2:.4f}, {3:.4f}) (w3, grad3):({4:.4f}, {5:.4f})".
              format(self.w1.item(), self.w1.grad.item() if self.w1.grad is not None else 0,
                     self.w2.item(), self.w2.grad.item() if self.w2.grad is not None else 0,
                     self.w3.item(), self.w3.grad.item() if self.w3.grad is not None else 0))

        return total_loss

    def dice_bce_edge_loss(self, pred, groundtruth):
        # 设置边界检测卷积核
        laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3, 3).cuda()

        # 二值化prediction
        boundary_pred = pred.detach().clone()
        boundary_pred[boundary_pred > 0.5] = 1
        boundary_pred[boundary_pred <= 0.5] = 0

        # 用拉普拉斯卷积核提取gt的边界信息
        boundary_pred = F.conv3d(boundary_pred, laplacian_kernel, padding=1)
        boundary_groundtruth = F.conv3d(groundtruth, laplacian_kernel, padding=1)

        # 这个值指只会大于0，被clamp在[min,]
        boundary_pred = boundary_pred.clamp(min=0)
        boundary_pred[boundary_pred > 0.1] = 1
        boundary_pred[boundary_pred <= 0.1] = 0
        boundary_groundtruth = boundary_groundtruth.clamp(min=0)
        boundary_groundtruth[boundary_groundtruth > 0.1] = 1
        boundary_groundtruth[boundary_groundtruth <= 0.1] = 0

        # 用最邻近插值使得边界gt与input对齐
        if pred.shape[-1] != boundary_groundtruth.shape[-1]:
            boundary_groundtruth = F.interpolate(
                boundary_groundtruth, pred.shape[2:], mode='nearest')

        # return (self.dice_edge_loss(pred, boundary_groundtruth),
        #         F.binary_cross_entropy(pred, boundary_groundtruth))
        return (self.dice_edge_loss(boundary_pred, boundary_groundtruth),
                F.binary_cross_entropy(boundary_pred, boundary_groundtruth))

    def cldice_loss(self, pred, groundtruth):
        skel_pred = self.soft_skel(pred, self.iter)
        skel_true = self.soft_skel(groundtruth, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, groundtruth)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, pred)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        return 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_skel(self, img, iter_):
        img1 = self.soft_dilate(self.soft_erode(img))
        skel = F.relu(img - img1)
        for j in range(iter_):
            img = self.soft_erode(img)
            img1 = self.soft_dilate(self.soft_erode(img))
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class AdaRegSpecLoss(nn.Module):
    """
    Reference to url link: https://ieeexplore.ieee.org/document/10163830
    called *Adaptive Regional Specific Loss*

    """

    def __init__(self, partition_size=16):
        super(AdaRegSpecLoss, self).__init__()
        self.pz = partition_size
        self.a = nn.Parameter(torch.Tensor([0.3]).cuda().requires_grad_())
        self.b = nn.Parameter(torch.Tensor([0.4]).cuda().requires_grad_())
        self.smooth = 1e-8

    def forward(self, pred, groundtruth):
        pred_final = torch.Tensor([]).cuda()
        gt_final = torch.Tensor([]).cuda()

        # 计算小块在各个维度的总数
        num_N = groundtruth.size()[0]
        num_z = groundtruth.size()[2] // self.pz
        num_x = groundtruth.size()[3] // self.pz
        num_y = groundtruth.size()[4] // self.pz

        # 将tensor分割成M个16*16*16的小块，即维度为(M, 16, 16, 16)
        for NN in range(num_N):
            gt_plane = torch.split(groundtruth[NN, :, :, :, :], self.pz, 1)
            pred_plane = torch.split(pred[NN, :, :, :, :], self.pz, 1)
            for zz in range(num_z):
                gt_line = torch.split(gt_plane[zz], self.pz, 2)
                pred_line = torch.split(pred_plane[zz], self.pz, 2)
                for xx in range(num_x):
                    gt_point = torch.split(gt_line[xx], self.pz, 3)
                    pred_point = torch.split(pred_line[xx], self.pz, 3)
                    for yy in range(num_y):
                        gt_final = torch.cat((gt_final, gt_point[yy]), dim=0)
                        pred_final = torch.cat((pred_final, pred_point[yy]), dim=0)

        # 将所有小块拉成向量 (M, 16*16*16)
        pred_final = torch.flatten(pred_final, start_dim=1, end_dim=-1)
        gt_final = torch.flatten(gt_final, start_dim=1, end_dim=-1)

        # 计算需要用的到confusing matrix数据TP, FN 和 FP
        tp = torch.sum(gt_final * pred_final, dim=1)
        fn = torch.sum(gt_final * (1 - pred_final), dim=1)
        fp = torch.sum((1 - gt_final) * pred_final, dim=1)

        # 计算自适应参数
        alpha = self.a + self.b * ((fp + self.smooth) / (fp + fn + self.smooth))
        beta = self.a + self.b * ((fn + self.smooth) / (fp + fn + self.smooth))

        # 加和Loss
        loss = torch.sum(1. - (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth))

        return loss


class AdaptiveRegionalEdgeDiceCLDiceLoss(nn.Module):
    """ Loss for Ours, (CenterLine + Edge + Dice) Reinforce Loss

    Input is expected to be probabilities instead of logits, and groundtruth should be normalized.

    """

    def __init__(self, threshold=0.8, partition_size=16):
        super(AdaptiveRegionalEdgeDiceCLDiceLoss, self).__init__()

        self.iter_cnt = 0
        self.tb_writer_weight = SummaryWriter(log_dir='./logs/0_LAC_logs/' + time.strftime('%Y%m%d_%H%M%S',
                                                                                           time.localtime(time.time())))

        self.dice_coeff = DiceCoefficient()

        # 总体参数
        self.weight0 = torch.tensor([0.5])  # 总Loss的权重
        self.dice_loss = DiceLoss()
        self.dice_edge_loss = DiceLoss()
        self.bce_edge_loss = nn.BCELoss()
        self.threshold = threshold

        # 局部参数
        self.w1 = nn.Parameter(torch.Tensor([1.]))  # 局部edgeloss的权重
        self.w2 = nn.Parameter(torch.Tensor([1.]))  # 局部clloss的权重
        # self.a = nn.Parameter(torch.Tensor([0.5]).cuda().requires_grad_())
        # self.b = nn.Parameter(torch.Tensor([0.5]).cuda().requires_grad_())
        self.a = torch.tensor([0.5])
        self.b = torch.tensor([0.5])

        self.pz = partition_size
        self.iter = 3
        self.smooth = 1.
        self.adp_smooth = 1e-8

    def forward(self, pred, groundtruth):
        # 计算Dice作为阈值判断标准
        dice_coeff = self.dice_coeff(pred, groundtruth)
        # 计算部分loss
        dice_loss = self.dice_loss(pred, groundtruth)
        # 整合Loss
        if dice_coeff.float() < self.threshold:
            total_loss = dice_loss
        else:
            total_loss = dice_loss + self.adaptive_regional_edge_cl_loss(pred, groundtruth)

        self.iter_cnt += 1
        return total_loss

    def adaptive_regional_edge_cl_loss(self, pred, groundtruth):

        pre_bdrs = torch.Tensor([]).cuda()
        gt_bdrs = torch.Tensor([]).cuda()
        pre_cls = torch.Tensor([]).cuda()
        gt_cls = torch.Tensor([]).cuda()

        # Blocks in every dimensions
        num_N = groundtruth.size()[0]
        num_z = groundtruth.size()[2] // self.pz
        num_x = groundtruth.size()[3] // self.pz
        num_y = groundtruth.size()[4] // self.pz

        # M total blocks
        M = num_N * num_z * num_x * num_y

        # Split into M (16*16*16) blocks. Dimension: (M, 16, 16, 16)
        for NN in range(num_N):
            gt_plane = torch.split(groundtruth[NN, :, :, :, :], self.pz, 1)  # (C, Z, X ,Y)
            pred_plane = torch.split(pred[NN, :, :, :, :], self.pz, 1)
            for zz in range(num_z):
                gt_line = torch.split(gt_plane[zz], self.pz, 2)
                pred_line = torch.split(pred_plane[zz], self.pz, 2)
                for xx in range(num_x):
                    gt_point = torch.split(gt_line[xx], self.pz, 3)
                    pred_point = torch.split(pred_line[xx], self.pz, 3)
                    for yy in range(num_y):
                        pred_bdr, gt_bdr = self.bdr(pred_point[yy], gt_point[yy])
                        pre_bdrs = torch.cat((pre_bdrs, pred_bdr), dim=0)
                        gt_bdrs = torch.cat((gt_bdrs, gt_bdr), dim=0)

                        pred_cl, gt_cl = self.cl(pred_point[yy], gt_point[yy])
                        pre_cls = torch.cat((pre_cls, pred_cl), dim=0)
                        gt_cls = torch.cat((gt_cls, gt_cl), dim=0)

        # 将所有小块拉成向量 (M, 16*16*16)
        pre_bdrs = torch.flatten(pre_bdrs, start_dim=1, end_dim=-1)
        gt_bdrs = torch.flatten(gt_bdrs, start_dim=1, end_dim=-1)
        pre_cls = torch.flatten(pre_cls, start_dim=1, end_dim=-1)
        gt_cls = torch.flatten(gt_cls, start_dim=1, end_dim=-1)

        # 计算需要用的到confusing matrix数据TP, FN 和 FP
        tp_bdr = torch.sum(gt_bdrs * pre_bdrs, dim=1)
        fn_bdr = torch.sum(gt_bdrs * (1 - pre_bdrs), dim=1)
        fp_bdr = torch.sum((1 - gt_bdrs) * pre_bdrs, dim=1)

        tp_cl = torch.sum(gt_cls * pre_cls, dim=1)
        fn_cl = torch.sum(gt_cls * (1 - pre_cls), dim=1)
        fp_cl = torch.sum((1 - gt_cls) * pre_cls, dim=1)

        # 计算自适应参数
        self.a = self.a.cuda()
        self.b = self.b.cuda()
        alpha_bdr = self.a + self.b * ((fp_bdr + self.adp_smooth) / (fp_bdr + fn_bdr + self.adp_smooth))
        beta_bdr = self.a + self.b * ((fn_bdr + self.adp_smooth) / (fp_bdr + fn_bdr + self.adp_smooth))
        alpha_cl = self.a + self.b * ((fp_cl + self.adp_smooth) / (fp_cl + fn_cl + self.adp_smooth))
        beta_cl = self.a + self.b * ((fn_cl + self.adp_smooth) / (fp_cl + fn_cl + self.adp_smooth))

        # 加和Loss
        loss_bdr = torch.sum(
            1. - (tp_bdr + self.adp_smooth) / (tp_bdr + alpha_bdr * fp_bdr + beta_bdr * fn_bdr + self.adp_smooth))
        loss_cl = torch.sum(
            1. - (tp_cl + self.adp_smooth) / (tp_cl + alpha_cl * fp_cl + beta_cl * fn_cl + self.adp_smooth))

        # 混合Loss
        loss = (self.w1.pow(-2) * loss_bdr + self.w2.pow(-2) * loss_cl) / (2 * M) + (
                1 + self.w1.abs() * self.w2.abs()).log()

        self.tb_writer_weight.add_scalars('Train_Loss_Trainable_Paras',
                                          {'L_e': self.w1.item(),
                                           'L_e_grad': self.w1.grad.item() if self.w1.grad is not None else 0,
                                           'L_cl': self.w2.item(),
                                           'L_cl_grad': self.w2.grad.item() if self.w1.grad is not None else 0},
                                          self.iter_cnt)

        # print("(w1, grad1):({0:.4f}, {1:.4f})  (w2, grad2):({2:.4f}, {3:.4f})".
        #       format(self.w1.item(), self.w1.grad.item() if self.w1.grad is not None else 0,
        #              self.w2.item(), self.w2.grad.item() if self.w2.grad is not None else 0))

        return loss

    def bdr(self, pred, groundtruth):
        # 设置边界检测卷积核
        laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3, 3).cuda()

        # 用拉普拉斯卷积核提取gt的边界信息
        boundary_pred = F.conv3d(pred, laplacian_kernel, padding=1)
        boundary_groundtruth = F.conv3d(groundtruth, laplacian_kernel, padding=1)
        # 这个值指只会大于0，被clamp在[min,]
        boundary_pred = boundary_pred.clamp(min=0)
        boundary_pred[boundary_pred > 0.1] = 1
        boundary_pred[boundary_pred <= 0.1] = 0
        boundary_groundtruth = boundary_groundtruth.clamp(min=0)
        boundary_groundtruth[boundary_groundtruth > 0.1] = 1
        boundary_groundtruth[boundary_groundtruth <= 0.1] = 0

        # 用最邻近插值使得边界gt与input对齐
        if pred.shape[-1] != boundary_groundtruth.shape[-1]:
            boundary_groundtruth = F.interpolate(
                boundary_groundtruth, pred.shape[2:], mode='nearest')

        return boundary_pred, boundary_groundtruth

    def cl(self, pred, groundtruth):
        cl_pred = self.soft_skel(pred, self.iter)
        cl_groundtruth = self.soft_skel(groundtruth, self.iter)

        return cl_pred, cl_groundtruth

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_skel(self, img, iter_):
        img1 = self.soft_dilate(self.soft_erode(img))
        skel = F.relu(img - img1)
        for j in range(iter_):
            img = self.soft_erode(img)
            img1 = self.soft_dilate(self.soft_erode(img))
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


# 创建Loss常用到的方法
#######################################################################################################################
def get_loss_criterion(loss_config):
    """
    Returns the loss function based on provided configuration
    :param loss_config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function

    """
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        weight = torch.tensor(weight)

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


def create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'DiceLoss':
        return DiceLoss(loss_config)
    elif name == 'EdgeReinforceLoss':
        # 需要在配置文件中添加er_threshold，默认为0.8
        threshold = loss_config.get('threshold')
        return EdgeReinforceLoss(threshold)
    elif name == 'BCELoss':
        return BCELoss()
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'EdgeDiceCLDiceLoss':
        # 需要在配置文件中添加er_threshold，默认为0.8
        threshold = loss_config.get('threshold')
        return EdgeDiceCLDiceLoss(threshold)
    elif name == 'AdaRegSpecLoss':
        partition_size = loss_config.get('partition_size')
        return AdaRegSpecLoss(partition_size)
    elif name == 'AdaptiveRegionalEdgeDiceCLDiceLoss':
        threshold = loss_config.get('threshold')
        partition_size = loss_config.get('partition_size')
        return AdaptiveRegionalEdgeDiceCLDiceLoss(threshold, partition_size)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
