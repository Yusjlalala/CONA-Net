import torch
import importlib
import numpy as np
from medpy import metric
from skimage.morphology import skeletonize, skeletonize_3d

class DiceCoefficient:
    """
    计算所有通道的平均Dice Coefficient.
    这里是训练用的，输入的是概率，一定不要做二值化处理！！！
    训练的时候用DiceLoss时，不要用Dice作为评估标准！！ otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, pred, groundtruth, is_train=None):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(pred, groundtruth, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, pred, groundtruth):
        """
        :param pred: 5D probability maps torch float tensor (NxCxDxHxW)
        :param groundtruth: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert pred.dim() == 5

        n_classes = pred.size()[1]

        if groundtruth.dim() == 4:
            groundtruth = get_one_hot(groundtruth, classes=n_classes, ignore_index=self.ignore_index)

        assert pred.size() == groundtruth.size()

        per_batch_iou = []
        for _input, _target in zip(pred, groundtruth):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, pred, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = pred > 0.5
            return result.long()

        _, max_index = torch.max(pred, dim=0, keepdim=True)
        return torch.zeros_like(pred, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, pred, groundtruth):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(pred & groundtruth).float() / torch.clamp(torch.sum(pred | groundtruth).float(), min=1e-8)


class BinaryMetrics:
    def __init__(self, voxel_spacing=None):
        """ For Binary classification
        输入均经过二值化处理，二分类的metrics直接全部计算完了

        Calculate [Dice], [HD95], [ASD], [Recall], [Sen], [Spec], [JC]

        Call args:
            pred(tensor)，target(tensor)
        """

        # if voxel_spacing is None:
        #     raise Exception(
        #         'Exception in BinaryMetrics call, [Voxel Spacing] is needed calculate HD95 and ASD!')
        self.voxel_spacing = voxel_spacing

    def __call__(self, pred, groundtruth, is_train):

        assert is_train is not None, "Please check the evaluation metrics process in 'Train Mode' or 'Prediction Mode'!"
        pred = pred.data.cpu().numpy()
        groundtruth = groundtruth.data.cpu().numpy()

        # for single channel input just threshold the probability map
        pred = np.where(pred > 0.5, 1., 0)

        if is_train:
            return metric.binary.dc(pred, groundtruth)
        else:
            # 测试集推理过程需要变成3个维度
            pred = pred.squeeze(0).squeeze(0)
            groundtruth = groundtruth.squeeze(0).squeeze(0)
            if len(pred.shape) != 3 and len(groundtruth.shape) != 3:
                raise Exception('Dimension should be 3')

            dice = metric.binary.dc(pred, groundtruth)
            cldice = clDice(pred, groundtruth)
            hd95 = metric.binary.hd95(pred, groundtruth, voxelspacing=self.voxel_spacing)
            asd = metric.binary.asd(pred, groundtruth, voxelspacing=self.voxel_spacing)
            # hd95=0.
            # asd=0.
            sen = metric.binary.sensitivity(pred, groundtruth)
            spec = metric.binary.specificity(pred, groundtruth)

            return dice, cldice, hd95, asd, sen, spec


# 创建metrics常用到的方法
#######################################################################################################################
def compute_per_channel_dice(pred, groundtruth, epsilon=1e-6, weight=None):
    """
    注意在加载数据的时候需要归一化才可以使用此公式，对于单通道的医学图像来说channel_dice实际上也是batch_dice。

    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel logit and
    groundtruth.

    Args:
         pred (torch.Tensor): NxCxSpatial pred tensor
         groundtruth (torch.Tensor): NxCxSpatial groundtruth tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert pred.size() == groundtruth.size(), "'input' and 'groundtruth' must have the same shape"

    pred = flatten(pred)
    groundtruth = flatten(groundtruth)
    groundtruth = groundtruth.float()

    # compute per channel Dice Coefficient
    intersect = (pred * groundtruth).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (pred * pred).sum(-1) + (groundtruth * groundtruth).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def clDice(v_p, v_g):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_g ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = 0.
    tsens = 0.
    if len(v_p.shape) == 2:
        tprec = np.sum(v_p * skeletonize(v_g)) / np.sum(skeletonize(v_g))
        tsens = np.sum(v_g * skeletonize(v_p)) / np.sum(skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = np.sum(v_p * skeletonize_3d(v_g)) / np.sum(skeletonize_3d(v_g))
        tsens = np.sum(v_g * skeletonize_3d(v_p)) / np.sum(skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec + tsens)


def get_one_hot(input, classes, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        classes (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = classes

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def get_evaluation_metric(eval_metric_config):
    """
    Returns the evaluation metric function based on provided configuration
    :param eval_metric_config: (dict) a 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def metric_class(class_name):
        m = importlib.import_module('metrics')
        clazz = getattr(m, class_name)
        return clazz

    metric_name = eval_metric_config.pop('name')
    metric_class = metric_class(metric_name)

    return metric_class(**eval_metric_config)
