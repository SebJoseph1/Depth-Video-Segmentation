import random

import numpy as np
import torch
import torch.nn as nn

import dataloader
from depthloss import DepthSmoothLoss, PanopticGuidedEdgeLoss, PanopticGuidedSmooth
from utils import log_to_json
from new_depth import DepthLoss


class L1Loss(nn.Module):
    """This class contains code to compute the top-k loss."""

    def __init__(self, weight=1.0, top_k_percent_pixels=1.0):
        """Initializes a top-k L1 loss.

        Args:
            weight: A float representing the weight of the loss.
            top_k_percent_pixels: An optional float specifying the percentage of
                pixels used to compute the loss. The value must lie within [0.0, 1.0].
        """
        super(L1Loss, self).__init__()

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)

        self.loss_function = mean_absolute_error
        self.top_k_percent_pixels = top_k_percent_pixels
        self.weight = weight

    def forward(self, y_true, y_pred, weights):
        """Computes the top-k loss.

        Args:
            y_true: A dict of tensors providing ground-truth information.
            y_pred: A dict of tensors providing predictions.
            weights: A tensor containing the weights.

        Returns:
            A tensor of shape [batch] containing the loss per sample.
        """
        per_pixel_loss = self.loss_function(y_true, y_pred)
        per_pixel_loss = torch.mul(per_pixel_loss, weights)

        return compute_average_top_k_loss(per_pixel_loss, self.top_k_percent_pixels) * self.weight


def mean_absolute_error(y_true, y_pred, force_keep_dims=False):
    """Computes the per-pixel mean absolute error for 3D and 4D tensors.

    Default reduction behavior: If a 3D tensor is used, no reduction is applied.
    In case of a 4D tensor, reduction is applied. This behavior can be overridden
    by force_keep_dims.

    Args:
        y_true: A tensor of shape [batch, height, width] or [batch, height,
            width, channels] containing the ground-truth.
        y_pred: A tensor of shape [batch, height, width] or [batch, height,
            width, channels] containing the prediction.
        force_keep_dims: A boolean flag specifying whether no reduction should be
            applied.

    Returns:
        A tensor with the mean absolute error.
    """
    assert y_pred.dim() in [3, 4], 'Input tensors must have rank 3 or 4.'
    if len(y_pred.shape) == 3 or force_keep_dims:
        return torch.abs(y_true - y_pred)
    else:
        return torch.mean(torch.abs(y_true - y_pred), dim=[1],keepdim= True)


def compute_average_top_k_loss(loss, top_k_percentage):
    """Computes the average top-k loss per sample.

    Args:
        loss: A tensor with 2 or more dimensions of shape [batch, ...].
        top_k_percentage: A float representing the % of pixel that should be used
            for calculating the loss.

    Returns:
        A tensor of shape [batch] containing the mean top-k loss per sample. Due to
        the use of different tf.strategy, we return the loss per sample and require
        explicit averaging by the user.
    """
    loss = loss.view(loss.size(0), -1)

    if top_k_percentage != 1.0:
        num_elements_per_sample = loss.size(1)
        top_k_pixels = int(round(top_k_percentage * num_elements_per_sample))

        def top_k_1d(inputs):
            return torch.topk(inputs, top_k_pixels, dim=-1)[0]

        loss = torch.stack([top_k_1d(sample) for sample in loss])

    # Compute mean loss over spatial dimension.
    # num_non_zero = torch.sum(torch.ne(loss, 0.0).float(), dim=1)
    # loss_sum_per_sample = torch.sum(loss, dim=1)
    # loss_per_sample = torch.div(loss_sum_per_sample, num_non_zero)
    # ret = torch.where(torch.isinf(loss_per_sample), torch.tensor(0.0), loss_per_sample)
    # return torch.where(torch.isnan(ret), torch.tensor(0.0), ret)

    return loss.mean()

    # When dividing two numbers A, B in a loss function L = A / B,
    # always make sure that the divisor B cannot be zero or close to zero.
    # I.e.: L = A / B.clamp(1e-6)

    # num_nonzero = (loss > 0.0).sum()
    # loss_per_sample = 

def mean_squared_error(y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       force_keep_dims=False, 
                       ignore_label: [None, float] = None) -> torch.Tensor:
    """Computes the per-pixel mean squared error for 3D and 4D tensors.

    Default reduction behavior: If a 3D tensor is used, no reduction is applied.
    In case of a 4D tensor, reduction is applied. This behavior can be overridden
    by force_keep_dims.

    Args:
        y_true: A torch.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the ground-truth.
        y_pred: A torch.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the prediction.
        force_keep_dims: A boolean flag specifying whether no reduction should be
        applied.

    Returns:
        A torch.Tensor with the mean squared error.
    """
    assert y_pred.dim() in [3, 4], 'Input tensors must have rank 3 or 4.'
    if ignore_label is not None:
        y_pred[y_true == ignore_label] = ignore_label
    if y_pred.dim() == 3 or force_keep_dims:
        return torch.square(y_true - y_pred)
    else:
        return torch.mean(torch.square(y_true - y_pred), dim=[1], keepdim=True)

class MSELoss(torch.nn.Module):
    """This class contains code to compute the top-k loss."""

    def __init__(self,
                 weight: float = 1.0,
                 top_k_percent_pixels: float = 1.0,
                 ignore_label: [None, float] = None):
        """Initializes a top-k L1 loss.

        Args:
            loss_function: A callable loss function.
            gt_key: A key to extract the ground-truth tensor.
            pred_key: A key to extract the prediction tensor.
            weight_key: A key to extract the weight tensor.
            top_k_percent_pixels: An optional float specifying the percentage of
            pixels used to compute the loss. The value must lie within [0.0, 1.0].
        """
        super(MSELoss, self).__init__()

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)

        self._loss_function = mean_squared_error
        self._top_k_percent_pixels = top_k_percent_pixels
        self._weight = weight
        self._ignore_label = ignore_label

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Computes the top-k loss.

        Args:
            y_true: A dict of tensors providing ground-truth information.
            y_pred: A dict of tensors providing predictions.

        Returns:
            A tensor of shape [batch] containing the loss per sample.
        """

        per_pixel_loss = self._loss_function(y_true, y_pred, ignore_label=self._ignore_label)
        per_pixel_loss = torch.mul(per_pixel_loss, weights)

        return compute_average_top_k_loss(per_pixel_loss,
                                          self._top_k_percent_pixels) * self._weight

import torch.nn.functional as F


def strided_downsample(input_tensor, target_size, input_size):
    """Strided downsamples a tensor to the target size.
  
    The stride_height and stride_width is computed by (height - 1) //
    (target_height - 1) and (width - 1) // (target_width - 1). We raise an error
    if stride_height != stride_width, since this is not intended in our current
    use cases. But this check can be removed if different strides are desired.
    This function supports static shape only.
  
    Args:
        input_tensor: A [batch, height, width] torch.Tensor to be downsampled.
        target_size: A list of two integers, [target_height, target_width], the
            target size after downsampling.
  
    Returns:
        output_tensor: A [batch, target_height, target_width] torch.Tensor, the
            downsampled result.
  
    Raises:
        ValueError: If the input cannot be downsampled with integer stride, i.e.,
            (height - 1) % (target_height - 1) != 0, or (width - 1) % (target_width -
            1) != 0.
        ValueError: If the height axis stride does not equal to the width axis
            stride.
    """
    input_height, input_width = input_size[1:3]
    target_height, target_width = target_size
  
    if ((input_height - 1) % (target_height - 1) or
        (input_width - 1) % (target_width - 1)):
        raise ValueError('The input cannot be downsampled with integer striding. '
                         'Please ensure (height - 1) % (target_height - 1) == 0 '
                         'and (width - 1) % (target_width - 1) == 0.')
  
    stride_height = (input_height - 1) // (target_height - 1)
    stride_width = (input_width - 1) // (target_width - 1)
  
    if stride_height != stride_width:
        raise ValueError('The height axis stride does not equal to the width axis '
                         'stride.')
  
    if stride_height > 1 or stride_width > 1:
        return input_tensor[:, ::stride_height, ::stride_width]
  
    return input_tensor

def encode_one_hot(gt, num_classes, weights, ignore_label):
    """Helper function for one-hot encoding of integer labels.
  
    Args:
        gt: A torch.Tensor providing ground-truth information. Integer type label.
        num_classes: An integer indicating the number of classes considered in the
            ground-truth. It is used as 'depth' in torch.nn.functional.one_hot().
        weights: A torch.Tensor containing weights information.
        ignore_label: An integer specifying the ignore label or None.
  
    Returns:
        gt: A torch.Tensor of one-hot encoded gt labels.
        weights: A torch.Tensor with ignore_label considered.
    """
    if ignore_label is not None:
        keep_mask = (gt != ignore_label).float()
        unkeep_mask = (gt == ignore_label)
    else:
        keep_mask = torch.ones_like(gt, dtype=torch.float32)
        unkeep_mask = torch.ones_like(gt, dtype=torch.bool)
    gt[unkeep_mask] = 19
    gt = torch.nn.functional.one_hot(gt.squeeze(1).long(), num_classes)
    gt = gt.permute(0, 3, 1, 2)
    gt = gt.float()
    weights = weights * keep_mask
    return gt, weights


def is_one_hot(gt, pred):
    """Helper function for checking if gt tensor is one-hot encoded or not.
  
    Args:
        gt: A torch.Tensor providing ground-truth information.
        pred: A torch.Tensor providing prediction information.
  
    Returns:
        A boolean indicating whether the gt is one-hot encoded (True) or
        in integer type (False).
    """
    gt_shape = gt.size()
    pred_shape = pred.size()
  
    # If the ground truth is one-hot encoded, the rank of the ground truth should
    # match that of the prediction. In addition, we check that the first
    # dimension, batch_size, and the last dimension, channels, should also match
    # the prediction. However, we still allow spatial dimensions, e.g., height and
    # width, to be different since we will downsample the ground truth if needed.
    return (len(gt_shape) == len(pred_shape) and
            gt_shape[0] == pred_shape[0] and gt_shape[1] == pred_shape[1])

class TopKCrossEntropyLoss(torch.nn.Module):
    """This class contains code for top-k cross-entropy."""

    def __init__(self, num_classes=None, ignore_label=None, top_k_percent_pixels=1.0, weight=1.0):
        """Initializes a top-k cross entropy loss.
  
        Args:
            gt_key: A key to extract the ground-truth tensor.
            pred_key: A key to extract the prediction tensor.
            weight_key: A key to extract the weight tensor.
            num_classes: An integer specifying the number of classes in the dataset.
            ignore_label: An optional integer specifying the ignore label or None.
            top_k_percent_pixels: An optional float specifying the percentage of
                pixels used to compute the loss. The value must lie within [0.0, 1.0].
            dynamic_weight: A boolean indicating whether the weights are determined
                dynamically w.r.t. the class confidence of each predicted mask.
  
        Raises:
            ValueError: An error occurs when top_k_percent_pixels is not between 0.0
                and 1.0.
        """
        super(TopKCrossEntropyLoss, self).__init__()

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)
  
        self.num_classes = num_classes
        self.ignore_label = torch.tensor(ignore_label)
        self.top_k_percent_pixels = top_k_percent_pixels
        self.weight = weight

    def forward(self, y_true, y_pred, weights):
        """Computes the top-k cross-entropy loss.
  
        Args:
            y_true: A dict of tensors providing ground-truth information. The tensors
                can be either integer type or one-hot encoded. When is integer type, the
                shape can be either [batch, num_elements] or [batch, height, width].
                When one-hot encoded, the shape can be [batch, num_elements, channels]
                or [batch, height, width, channels].
            y_pred: A dict of tensors providing predictions. The tensors are of shape
                [batch, num_elements, channels] or [batch, height, width, channels]. If
                the prediction is 2D (with height and width), we allow the spatial
                dimension to be strided_height and strided_width. In this case, we
                downsample the ground truth accordingly.
  
        Returns:
            A tensor of shape [batch] containing the loss per image.
  
        Raises:
            ValueError: If the prediction is 1D (with the length dimension) but its
                length does not match that of the ground truth.
        """
        gt_shape = y_true.size()
        gt_shape_comp = gt_shape[:1] + gt_shape[1+1:]

        pred_shape = y_pred.size()
        pred_shape_comp = pred_shape[:1] + pred_shape[1+1:]
  
        # y_pred, weights = encode_one_hot(y_pred, self.num_classes, weights, self.ignore_label)
        y_true = y_true.squeeze(1)
        # # Downsample the ground truth for 2D prediction cases.
        # if len(pred_shape) == 4 and gt_shape_comp[1:3] != pred_shape_comp[1:3]:# haven't tested this
        #     y_true = strided_downsample(y_true, pred_shape_comp[1:3], gt_shape_comp)
        #     weights = strided_downsample(weights, pred_shape_comp[1:3], gt_shape_comp)
        # elif len(pred_shape) == 3 and gt_shape_comp[1] != pred_shape_comp[1]:
        #     # We don't support downsampling for 1D predictions.
        #     raise ValueError('The shape of gt does not match the shape of pred.')
  
        # if is_one_hot(y_true, y_pred):
        #     y_true = y_true.float()
        # else:
        #     y_true = y_true.int()
        #     y_true, weights = encode_one_hot(y_true, self.num_classes, weights, self.ignore_label)
  
        pixel_losses = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none',ignore_index=self.ignore_label)
        if torch.sum(torch.isnan(pixel_losses)) > 0:
            op = 0
        weighted_pixel_losses = pixel_losses.unsqueeze(1) * weights
        kio = compute_average_top_k_loss(weighted_pixel_losses, self.top_k_percent_pixels) * self.weight
        if torch.sum(torch.isnan(kio)) > 0:
            op = 0
        return compute_average_top_k_loss(weighted_pixel_losses, self.top_k_percent_pixels) * self.weight


class SILogError(nn.Module):
    """This class contains code to compute the SILog error.

    Scale invariant logarithmic (SILog) error was proposed for monocular depth
    estimation.

    Reference:
    Eigen, David, Christian Puhrsch, and Rob Fergus. "Depth map prediction from a
    single image using a multi-scale deep network." In NeurIPS, 2014.
    """

    def __init__(self, ignore_label):
        super(SILogError, self).__init__()
        self.ignore_label = torch.tensor(ignore_label)

    def forward(self, y_true, y_pred):
        """Computes the scale invariant logarithmic error.

        Args:
          y_true: A tensor providing ground-truth information.
          y_pred: A tensor providing predictions.

        Returns:
          A tensor containing the loss per sample.
        """

        def _calculate_error(y_true_sample, y_pred_sample):
            label_mask = y_true_sample != self.ignore_label
            y_true_sample = y_true_sample[label_mask]
            if len(y_true_sample) == 0:
                device = y_true_sample.get_device()#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.tensor(0.0).to(device)
            y_pred_sample = y_pred_sample[label_mask]
            # Scale invariant logarithmic error.
            gt_log = torch.log(y_true_sample)
            pred_log = torch.log(y_pred_sample)
            silog_error = (torch.mean(torch.square(gt_log - pred_log)) -
                           torch.square(torch.mean(gt_log - pred_log)))
            return silog_error

        silog_error = torch.stack(
            [_calculate_error(y_true_sample, y_pred_sample) for (y_true_sample, y_pred_sample) in zip(y_true, y_pred)])

        return silog_error


class RelativeSquaredError(nn.Module):
    """This class contains code to compute the relative squared error.

    This class computes the relative squared error for monocular depth estimation.

    Reference:
    Uhrig, Jonas, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and
    Andreas Geiger. "Sparsity invariant cnns." In 3DV, 2017.
    """

    def __init__(self, ignore_label):
        super(RelativeSquaredError, self).__init__()
        self.ignore_label = torch.tensor(ignore_label)

    def forward(self, y_true, y_pred):
        """Computes the relative squared error.

        Args:
          y_true: A tensor providing ground-truth information.
          y_pred: A tensor providing predictions.

        Returns:
          A tensor containing the loss per sample.
        """

        def _calculate_error(y_true_sample, y_pred_sample):
            label_mask = y_true_sample != self.ignore_label
            y_true_sample = y_true_sample[label_mask]
            if len(y_true_sample) == 0:
                device = y_true_sample.get_device()#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.tensor(0.0).to(device)
            y_pred_sample = y_pred_sample[label_mask]
            relative_squared_error = torch.sqrt(torch.mean(torch.square((y_true_sample - y_pred_sample) / y_true_sample)))
            return relative_squared_error

        relative_squared_error = torch.stack(
            [_calculate_error(y_true_sample, y_pred_sample) for (y_true_sample, y_pred_sample) in zip(y_true, y_pred)])
        return relative_squared_error


class SILogPlusRelativeSquaredLoss(nn.Module):
    """This class contains code to compute depth loss SILog + RelativeSquared.

    This depth loss function combines the scale invariant logarithmic (SILog)
    error and relative squared error, which was adopted in the ViP-DeepLab model.

    Reference:
    Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
    "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
    Segmentation." In CVPR 2021.
    """

    def __init__(self, ignore_label, weight=1.0):
        super(SILogPlusRelativeSquaredLoss, self).__init__()
        self.silog_error = SILogError(ignore_label)
        self.relative_squared_error = RelativeSquaredError(ignore_label)
        self.weight = weight

    def forward(self, y_true, y_pred):
        """Computes the loss for SILog + RelativeSquared.

        Args:
          y_true: A tensor providing ground-truth information.
          y_pred: A tensor providing predictions.

        Returns:
          A tensor containing the loss per sample.
        """
        return (self.silog_error(y_true, y_pred) + self.relative_squared_error(y_true, y_pred)) * self.weight



if __name__ == "__main__":
    pass
    #a = TopKCrossEntropyLoss(num_classes=10,ignore_label=0,top_k_percent_pixels=0.2)
    #a1 = torch.randint(0, 8 + 1, (3,1,73,567), dtype=torch.int)
    #a2 = torch.randn(3,10,73,567)
    #a3 = torch.randn(3,1,73,567)
    #loss = a(a1,a2,a3)


def sum_vip_deeplab_losses(images, outputs, labels):
        depth_loss = DepthLoss(weight=1.0).forward(labels['depth'].clone(), outputs['depth'].clone(), mask=torch.ones_like(outputs['depth']).bool())
        #depth_loss = MSELoss(weight=1.0, ignore_label=0.0).forward(labels['depth'].clone(), outputs['depth'].clone(), torch.ones_like(outputs['depth']))
        #depth_loss = SILogPlusRelativeSquaredLoss(0.0, 0.1).forward(labels['depth'], outputs['depth'])
        depth_smoothness_loss = DepthSmoothLoss(depth_max=89,eps=1e-6,weight=10.0).forward(images=images[:,:3,:,:].clone(),depths=outputs['depth'].clone())
        depth_edge_loss = PanopticGuidedEdgeLoss(depth_max=89,eps=1e-6,weight=4.0).forward(panoptic_truth=labels['panoptic'].clone(),depths=outputs['depth'].clone())
        depth_guided_smoothness_loss = PanopticGuidedSmooth(depth_max=89,eps=1e-6,weight=10.0).forward(panoptic_truth=labels['panoptic'].clone(),depths=outputs['depth'].clone())
        # print(loss)
        semantic_loss = TopKCrossEntropyLoss(num_classes=len(dataloader.classes), ignore_label=255, weight=1.0, top_k_percent_pixels=0.5).forward(labels['semantic'].clone(), outputs['semantic'].clone().float(), labels['semantic_weights'])
        # print(loss)
        center_loss = MSELoss(weight=200.0).forward(labels['center'].clone(), outputs['center'].clone(), labels['semantic_weights'])
        # print(loss)
        offset_loss = L1Loss(weight=0.05).forward(labels['offset'].clone(), outputs['offset'].clone(), labels['semantic_weights'])
        # print(loss)
        nxtoffset_loss = L1Loss(weight=0.05).forward(labels['nxtoffset'].clone(), outputs['nxtoffset'].clone(), labels['semantic_weights'])
        # print(loss)
        loss = depth_loss + semantic_loss + center_loss + offset_loss + nxtoffset_loss + depth_edge_loss + depth_guided_smoothness_loss + depth_smoothness_loss
        batch_loss = torch.mean(loss)
        # print("depth ",depth_loss,"semanticloss ",semantic_loss," center ",center_loss," offset ",offset_loss," nxtoffset ",nxtoffset_loss)
        log_to_json('all_losses_output', {'depth_loss': torch.mean(depth_loss).item(), 'depth_smoothness_loss': torch.mean(depth_smoothness_loss).item(), 'depth_edge_loss': torch.mean(depth_edge_loss).item(), 'depth_guided_smoothness_loss': torch.mean(depth_guided_smoothness_loss).item(), 'semantic_loss': torch.mean(semantic_loss).item(), 'center_loss': torch.mean(center_loss).item(), 'offset_loss': torch.mean(offset_loss).item(), 'next_offset_loss': torch.mean(nxtoffset_loss).item()})
        return batch_loss 

