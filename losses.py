from typing import Text, Dict, Callable, Optional

import tensorflow as tf
import torch


def compute_average_top_k_loss(loss: tf.Tensor,
                               top_k_percentage: float) -> tf.Tensor:
    """Computes the avaerage top-k loss per sample.

    Args:
      loss: A tf.Tensor with 2 or more dimensions of shape [batch, ...].
      top_k_percentage: A float representing the % of pixel that should be used
        for calculating the loss.

    Returns:
      A tensor of shape [batch] containing the mean top-k loss per sample. Due to
      the use of different tf.strategy, we return the loss per sample and require
      explicit averaging by the user.
    """
    loss = tf.reshape(loss, shape=(tf.shape(loss)[0], -1))

    if top_k_percentage != 1.0:
        num_elements_per_sample = tf.shape(loss)[1]
        top_k_pixels = tf.cast(
            tf.math.round(top_k_percentage *
                          tf.cast(num_elements_per_sample, tf.float32)), tf.int32)

        def top_k_1d(inputs):
            return tf.math.top_k(inputs, top_k_pixels, sorted=False)[0]

        loss = tf.map_fn(fn=top_k_1d, elems=loss)

    # Compute mean loss over spatial dimension.
    num_non_zero = tf.reduce_sum(tf.cast(tf.not_equal(loss, 0.0), tf.float32), 1)
    loss_sum_per_sample = tf.reduce_sum(loss, 1)
    return tf.math.divide_no_nan(loss_sum_per_sample, num_non_zero)


def strided_downsample(input_tensor, target_size):
    """Strided downsamples a tensor to the target size.

    The stride_height and stride_width is computed by (height - 1) //
    (target_height - 1) and (width - 1) // (target_width - 1). We raise an error
    if stride_height != stride_width, since this is not intended in our current
    use cases. But this check can be removed if different strides are desired.
    This function supports static shape only.

    Args:
      input_tensor: A [batch, height, width] tf.Tensor to be downsampled.
      target_size: A list of two integers, [target_height, target_width], the
        target size after downsampling.

    Returns:
      output_tensor: A [batch, target_height, target_width] tf.Tensor, the
        downsampled result.

    Raises:
      ValueError: If the input cannot be downsampled with integer stride, i.e.,
        (height - 1) % (target_height - 1) != 0, or (width - 1) % (target_width -
        1) != 0.
      ValueError: If the height axis stride does not equal to the width axis
        stride.
    """
    input_height, input_width = input_tensor.get_shape().as_list()[1:3]
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


def encode_one_hot(gt: tf.Tensor,
                   num_classes: int,
                   weights: tf.Tensor,
                   ignore_label: Optional[int]):
    """Helper function for one-hot encoding of integer labels.

    Args:
      gt: A tf.Tensor providing ground-truth information. Integer type label.
      num_classes: An integer indicating the number of classes considered in the
        ground-truth. It is used as 'depth' in tf.one_hot().
      weights: A tf.Tensor containing weights information.
      ignore_label: An integer specifying the ignore label or None.

    Returns:
      gt: A tf.Tensor of one-hot encoded gt labels.
      weights: A tf.Tensor with ignore_label considered.
    """
    if ignore_label is not None:
        keep_mask = tf.cast(tf.not_equal(gt, ignore_label), dtype=tf.float32)
    else:
        keep_mask = tf.ones_like(gt, dtype=tf.float32)
    gt = tf.stop_gradient(tf.one_hot(gt, num_classes))
    weights = tf.multiply(weights, keep_mask)
    return gt, weights


def is_one_hot(gt: tf.Tensor, pred: tf.Tensor):
    """Helper function for checking if gt tensor is one-hot encoded or not.

    Args:
      gt: A tf.Tensor providing ground-truth information.
      pred: A tf.Tensor providing prediction information.

    Returns:
      A boolean indicating whether the gt is one-hot encoded (True) or
      in integer type (False).
    """
    gt_shape = gt.get_shape().as_list()
    pred_shape = pred.get_shape().as_list()
    # If the ground truth is one-hot encoded, the rank of the ground truth should
    # match that of the prediction. In addition, we check that the first
    # dimension, batch_size, and the last dimension, channels, should also match
    # the prediction. However, we still allow spatial dimensions, e.g., height and
    # width, to be different since we will downsample the ground truth if needed.
    return (len(gt_shape) == len(pred_shape) and
            gt_shape[0] == pred_shape[0] and gt_shape[-1] == pred_shape[-1])


class TopKCrossEntropyLoss(tf.keras.losses.Loss):
    """This class contains code for top-k cross-entropy."""

    def __init__(self,
                 num_classes: Optional[int],
                 ignore_label: Optional[int],
                 top_k_percent_pixels: float = 1.0,
                 weight: float = 1.0):
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
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super(TopKCrossEntropyLoss,
              self).__init__(reduction=tf.keras.losses.Reduction.NONE)

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)

        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._top_k_percent_pixels = top_k_percent_pixels
        self._weight = weight

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
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

        gt_shape = y_true.get_shape().as_list()
        pred_shape = y_pred.get_shape().as_list()

        # Downsample the ground truth for 2D prediction cases.
        if len(pred_shape) == 4 and gt_shape[1:3] != pred_shape[1:3]:
            y_true = strided_downsample(y_true, pred_shape[1:3])
            weights = strided_downsample(weights, pred_shape[1:3])
        elif len(pred_shape) == 3 and gt_shape[1] != pred_shape[1]:
            # We don't support downsampling for 1D predictions.
            raise ValueError('The shape of gt does not match the shape of pred.')

        if is_one_hot(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
        else:
            y_true = tf.cast(y_true, tf.int32)
            y_true, weights = encode_one_hot(y_true, self._num_classes, weights,
                                             self._ignore_label)
        pixel_losses = tf.keras.backend.categorical_crossentropy(
            y_true, y_pred, from_logits=True)
        weighted_pixel_losses = tf.multiply(pixel_losses, weights)  # hasn't understood

        return compute_average_top_k_loss(weighted_pixel_losses,
                                          self._top_k_percent_pixels) * self._weight


def mean_absolute_error(y_true: tf.Tensor,
                        y_pred: tf.Tensor,
                        force_keep_dims=False) -> tf.Tensor:
    """Computes the per-pixel mean absolute error for 3D and 4D tensors.

    Default reduction behavior: If a 3D tensor is used, no reduction is applied.
    In case of a 4D tensor, reduction is applied. This behavior can be overridden
    by force_keep_dims.
    Note: tf.keras.losses.mean_absolute_error always reduces the output by one
    dimension.

    Args:
      y_true: A tf.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the ground-truth.
      y_pred: A tf.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the prediction.
      force_keep_dims: A boolean flag specifying whether no reduction should be
        applied.

    Returns:
      A tf.Tensor with the mean absolute error.
    """
    tf.debugging.assert_rank_in(
        y_pred, [3, 4], message='Input tensors must have rank 3 or 4.')
    if len(y_pred.shape.as_list()) == 3 or force_keep_dims:
        return tf.abs(y_true - y_pred)
    else:
        return tf.reduce_mean(tf.abs(y_true - y_pred), axis=[3])


class L1Loss(tf.keras.losses.Loss):
    """This class contains code to compute the top-k loss."""

    def __init__(self,
                 weight: float = 1.0,
                 top_k_percent_pixels: float = 1.0):
        """Initializes a top-k L1 loss.

        Args:
          loss_function: A callable loss function.
          gt_key: A key to extract the ground-truth tensor.
          pred_key: A key to extract the prediction tensor.
          weight_key: A key to extract the weight tensor.
          top_k_percent_pixels: An optional float specifying the percentage of
            pixels used to compute the loss. The value must lie within [0.0, 1.0].
        """
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super(L1Loss,
              self).__init__(reduction=tf.keras.losses.Reduction.NONE)

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)

        self._loss_function = mean_absolute_error
        self._top_k_percent_pixels = top_k_percent_pixels
        self._weight = weight

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """Computes the top-k loss.

        Args:
          y_true: A dict of tensors providing ground-truth information.
          y_pred: A dict of tensors providing predictions.

        Returns:
          A tensor of shape [batch] containing the loss per sample.
        """

        per_pixel_loss = self._loss_function(y_true, y_pred)
        per_pixel_loss = tf.multiply(per_pixel_loss, weights)

        return compute_average_top_k_loss(per_pixel_loss,
                                          self._top_k_percent_pixels) * self._weight


def mean_squared_error(y_true: tf.Tensor,
                       y_pred: tf.Tensor,
                       force_keep_dims=False) -> tf.Tensor:
    """Computes the per-pixel mean squared error for 3D and 4D tensors.

    Default reduction behavior: If a 3D tensor is used, no reduction is applied.
    In case of a 4D tensor, reduction is applied. This behavior can be overridden
    by force_keep_dims.
    Note: tf.keras.losses.mean_squared_error always reduces the output by one
    dimension.

    Args:
      y_true: A tf.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the ground-truth.
      y_pred: A tf.Tensor of shape [batch, height, width] or [batch, height,
        width, channels] containing the prediction.
      force_keep_dims: A boolean flag specifying whether no reduction should be
        applied.

    Returns:
      A tf.Tensor with the mean squared error.
    """
    tf.debugging.assert_rank_in(
        y_pred, [3, 4], message='Input tensors must have rank 3 or 4.')
    if len(y_pred.shape.as_list()) == 3 or force_keep_dims:
        return tf.square(y_true - y_pred)
    else:
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=[3])


class MSELoss(tf.keras.losses.Loss):
    """This class contains code to compute the top-k loss."""

    def __init__(self,
                 weight: float = 1.0,
                 top_k_percent_pixels: float = 1.0):
        """Initializes a top-k L1 loss.

        Args:
          loss_function: A callable loss function.
          gt_key: A key to extract the ground-truth tensor.
          pred_key: A key to extract the prediction tensor.
          weight_key: A key to extract the weight tensor.
          top_k_percent_pixels: An optional float specifying the percentage of
            pixels used to compute the loss. The value must lie within [0.0, 1.0].
        """
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super(MSELoss,
              self).__init__(reduction=tf.keras.losses.Reduction.NONE)

        if top_k_percent_pixels < 0.0 or top_k_percent_pixels > 1.0:
            raise ValueError('The top-k percentage parameter must lie within 0.0 and '
                             '1.0, but %f was given' % top_k_percent_pixels)

        self._loss_function = mean_squared_error
        self._top_k_percent_pixels = top_k_percent_pixels
        self._weight = weight

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """Computes the top-k loss.

        Args:
          y_true: A dict of tensors providing ground-truth information.
          y_pred: A dict of tensors providing predictions.

        Returns:
          A tensor of shape [batch] containing the loss per sample.
        """

        per_pixel_loss = self._loss_function(y_true, y_pred)
        per_pixel_loss = tf.multiply(per_pixel_loss, weights)

        return compute_average_top_k_loss(per_pixel_loss,
                                          self._top_k_percent_pixels) * self._weight


class SILogError(tf.keras.losses.Loss):
    """This class contains code to compute the SILog error.

    Scale invariant logarithmic (SILog) error was proposed for monocular depth
    estimation.

    Reference:
    Eigen, David, Christian Puhrsch, and Rob Fergus. "Depth map prediction from a
    single image using a multi-scale deep network." In NeurIPS, 2014.
    """

    def __init__(self,
                 ignore_label: float):
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super().__init__(reduction='none')
        self._ignore_label = ignore_label

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor) -> tf.Tensor:
        """Computes the scale invariant logarithmic error.

        Args:
          y_true: A dict of tensors providing ground-truth information.
          y_pred: A dict of tensors providing predictions.

        Returns:
          A tensor of shape [batch] containing the loss per sample.
        """
        ignore_label = self._ignore_label

        def _compute_error(loss_input):
            y_true, y_pred = loss_input
            label_mask = y_true != ignore_label
            y_true = tf.boolean_mask(y_true, label_mask)
            y_pred = tf.boolean_mask(y_pred, label_mask)
            # Scale invariant logarithmic error.
            gt_log = tf.math.log(y_true)
            pred_log = tf.math.log(y_pred)
            silog_error = (tf.reduce_mean(tf.square(gt_log - pred_log)) -
                           tf.square(tf.reduce_mean(gt_log - pred_log)))
            return silog_error

        return tf.map_fn(_compute_error, (y_true, y_pred), fn_output_signature=tf.float32)


class RelativeSquaredError(tf.keras.losses.Loss):
    """This class contains code to compute the relative squared error.

    This class computes the relative squared error for monocular depth estimation.

    Reference:
    Uhrig, Jonas, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and
    Andreas Geiger. "Sparsity invariant cnns." In 3DV, 2017.
    """

    def __init__(self,
                 ignore_label: float):
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)
        self._ignore_label = ignore_label

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor) -> tf.Tensor:
        """Computes the relative squared error.

        Args:
          y_true: A dict of tensors providing ground-truth information.
          y_pred: A dict of tensors providing predictions.

        Returns:
          A tensor of shape [batch] containing the loss per sample.
        """
        ignore_label = self._ignore_label

        def _compute_error(loss_input):
            y_true, y_pred = loss_input
            label_mask = y_true != ignore_label
            y_true = tf.boolean_mask(y_true, label_mask)
            y_pred = tf.boolean_mask(y_pred, label_mask)
            # Relative squared error.
            relative_squared_error = tf.sqrt(
                tf.reduce_mean(tf.square((y_true - y_pred) / y_true)))
            return relative_squared_error

        return tf.map_fn(_compute_error, (y_true, y_pred), fn_output_signature=tf.float32)


class SILogPlusRelativeSquaredLoss(tf.keras.losses.Loss):
    """This class contains code to compute depth loss SILog + RelativeSquared.

    This depth loss function combines the scale invariant logarithmic (SILog)
    error and relative squared error, which was adopted in the ViP-DeepLab model.

    Reference:
    Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
    "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
    Segmentation." In CVPR 2021.
    """

    def __init__(self,
                 ignore_label: float,
                 weight: float = 1.0):
        # Implicit reduction might mess with tf.distribute.Strategy, hence we
        # explicitly reduce the loss.
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)
        self._silog_error = SILogError(ignore_label)
        self._relativate_squared_error = RelativeSquaredError(ignore_label)
        self._weight = weight

    def call(self, y_true: tf.Tensor,
             y_pred: tf.Tensor) -> tf.Tensor:
        """Computes the loss for SILog + RelativeSquared.

        Args:
          y_true: A dict of tensors providing ground-truth information.
          y_pred: A dict of tensors providing predictions.

        Returns:
          A tensor of shape [batch] containing the loss per sample.
        """
        return (self._silog_error(y_true, y_pred) + self._relativate_squared_error(
            y_true, y_pred)) * self._weight


class VIPDeepLabLossesSum(torch.nn.Module):

    def __init__(self):
        super(VIPDeepLabLossesSum, self).__init__()

    def forward(self, outputs, labels):

        loss = 0.0

        loss += depth_loss(outputs['depth'], labels['depath'])
        loss += cseloss(outputs['semantic'], labels['semantic'])
        #loss +=

        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)

        # mask out 'PAD' tokens
        mask = (labels >= 0).float()

        # the number of tokens is the sum of elements in mask
        num_tokens = int(torch.sum(mask).data[0])

        # pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels] * mask

        # cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs) / num_tokens