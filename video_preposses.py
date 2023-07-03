from typing import MutableMapping, Tuple

import numpy as np
import tensorflow as tf
import torch


class VideoPanopticPredictionStitcher():
    """The TF implementation of the stitching algorithm in ViP-DeepLab.

    It stitches a pair of image panoptic predictions to form video
    panoptic predictions by propagating instance IDs from concat_panoptic to
    next_panoptic based on IoU matching.

    Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
    "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
    Segmentation." CVPR, 2021.
    """

    def __init__(self,
                 label_divisor: int,
                 combine_offset: int = 2 ** 32, ):
        """Initializes a TF video panoptic prediction stitcher.

        It also sets the overlap_offset to label_divisor // 2 to avoid overlap
        between IDs in next_panoptic and the propagated IDs from concat_panoptic.
        label_divisor // 2 gives equal space for the two frames.

        Args:
          label_divisor: An integer specifying the label divisor of the dataset.
          combine_offset: An integer offset to combine concat and next panoptic.
          name: A string specifying the model name.
        """

        self._label_divisor = label_divisor
        self._overlap_offset = label_divisor // 2
        self._combine_offset = combine_offset

    def _ids_to_counts(
            self, id_array: tf.Tensor) -> tf.lookup.experimental.MutableHashTable:
        """Given a tf Tensor, returns a mapping from its elements to their counts.

        Args:
          id_array: A tf.Tensor from which the function counts its elements.

        Returns:
          A MutableHashTable that maps the elements to their counts.
        """
        ids, _, counts = tf.unique_with_counts(tf.reshape(id_array, [-1]))
        table = tf.lookup.experimental.MutableHashTable(
            key_dtype=ids.dtype, value_dtype=counts.dtype, default_value=-1)
        table.insert(ids, counts)
        return table

    def _increase_instance_ids_by_offset(self, panoptic: tf.Tensor) -> tf.Tensor:
        """Increases instance IDs by self._overlap_offset.

        Args:
          panoptic: A tf.Tensor for the panoptic prediction.

        Returns:
          A tf.Tensor for paonptic prediction with increased instance ids.
        """
        category = panoptic // self._label_divisor
        instance = panoptic % self._label_divisor
        # We skip 0 which is reserved for crowd.
        instance_mask = tf.greater(instance, 0)
        tf.assert_less(
            tf.reduce_max(instance + self._overlap_offset),
            self._label_divisor,
            message='Any new instance IDs cannot exceed label_divisor.')
        instance = tf.where(instance_mask, instance + self._overlap_offset,
                            instance)
        return category * self._label_divisor + instance

    def _compute_and_sort_iou_between_panoptic_ids(
            self, panoptic_1: tf.Tensor,
            panoptic_2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes and sorts intersecting panoptic IDs by IoU.

        Args:
          panoptic_1: A tf.Tensor for the panoptic prediction for frame 1.
          panoptic_2: A tf.Tensor for the panoptic prediction for frame 2.

        Returns:
          A tuple of tf.Tensor storing the mapping ids between the two input frames
            with their IoUs in the ascending order.
        """
        segment_areas_1 = self._ids_to_counts(panoptic_1)  # get ids and counts
        segment_areas_2 = self._ids_to_counts(panoptic_2)
        intersection_id_array = (
                tf.cast(panoptic_1, tf.int64) * self._combine_offset +
                tf.cast(panoptic_2, tf.int64))
        intersection_areas_table = self._ids_to_counts(intersection_id_array)
        intersection_ids, intersection_areas = intersection_areas_table.export()
        panoptic_ids_1 = tf.cast(intersection_ids // self._combine_offset, tf.int32)  # category
        panoptic_ids_2 = tf.cast(intersection_ids % self._combine_offset, tf.int32)  # id
        category_ids_1 = panoptic_ids_1 // self._label_divisor
        category_ids_2 = panoptic_ids_2 // self._label_divisor
        instance_ids_1 = panoptic_ids_1 % self._label_divisor
        instance_ids_2 = panoptic_ids_2 % self._label_divisor
        unions = (
                segment_areas_1.lookup(panoptic_ids_1) +
                segment_areas_2.lookup(panoptic_ids_2) - intersection_areas)
        intersection_ious = intersection_areas / unions
        is_valid_intersection = tf.logical_and(
            tf.equal(category_ids_1, category_ids_2),
            tf.logical_and(
                tf.not_equal(instance_ids_1, 0), tf.not_equal(instance_ids_2, 0)))
        intersection_ious = tf.gather(intersection_ious,
                                      tf.where(is_valid_intersection)[:, 0])
        panoptic_ids_1 = tf.gather(panoptic_ids_1,
                                   tf.where(is_valid_intersection)[:, 0])
        panoptic_ids_2 = tf.gather(panoptic_ids_2,
                                   tf.where(is_valid_intersection)[:, 0])
        ious_indices = tf.argsort(intersection_ious)
        panoptic_ids_1 = tf.gather(panoptic_ids_1, ious_indices)
        panoptic_ids_2 = tf.gather(panoptic_ids_2, ious_indices)
        segment_areas_1.remove(segment_areas_1.export()[0])
        segment_areas_2.remove(segment_areas_2.export()[0])
        intersection_areas_table.remove(intersection_areas_table.export()[0])
        return panoptic_ids_1, panoptic_ids_2

    def _match_and_propagate_instance_ids(
            self, concat_panoptic: tf.Tensor, next_panoptic: tf.Tensor,
            concat_panoptic_ids: tf.Tensor,
            next_panoptic_ids: tf.Tensor) -> tf.Tensor:
        """Propagates instance ids based on instance id matching.

        It propagates the instance ids from concat_panoptic to next_panoptic based
        on the mapping specified by concat_panoptic_ids and next_panoptic_ids.

        Args:
          concat_panoptic: A tf.Tensor for the concat panoptic prediction.
          next_panoptic: A tf.Tensor for the next panoptic prediction.
          concat_panoptic_ids: A tf.Tensor for the matching ids in concat_panoptic.
          next_panoptic_ids: A tf.Tensor for the matching ids in next_panoptic.

        Returns:
          A tf.Tensor for the next panoptic prediction with instance ids propagated
            from concat_panoptic.
        """
        map_concat_to_next = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int32, value_dtype=tf.int32, default_value=-1)
        map_concat_to_next.insert(
            tf.cast(concat_panoptic_ids, tf.int32),
            tf.cast(next_panoptic_ids, tf.int32))
        map_next_to_concat = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int32, value_dtype=tf.int32, default_value=-1)
        map_next_to_concat.insert(
            tf.cast(next_panoptic_ids, tf.int32),
            tf.cast(concat_panoptic_ids, tf.int32))
        matched_concat_panoptic_ids, matched_next_panoptic_ids = (
            map_concat_to_next.export())
        returned_concat_panoptic_ids = map_next_to_concat.lookup(
            matched_next_panoptic_ids)
        matched_ids_mask = tf.equal(matched_concat_panoptic_ids,
                                    returned_concat_panoptic_ids)
        matched_concat_panoptic_ids = tf.gather(matched_concat_panoptic_ids,
                                                tf.where(matched_ids_mask)[:, 0])
        matched_next_panoptic_ids = tf.gather(matched_next_panoptic_ids,
                                              tf.where(matched_ids_mask)[:, 0])
        matched_concat_panoptic_ids = tf.expand_dims(
            tf.expand_dims(matched_concat_panoptic_ids, axis=-1), axis=-1)
        matched_next_panoptic_ids = tf.expand_dims(
            tf.expand_dims(matched_next_panoptic_ids, axis=-1), axis=-1)
        propagate_mask = tf.equal(next_panoptic, matched_next_panoptic_ids)
        panoptic_to_replace = tf.reduce_sum(
            tf.where(propagate_mask, matched_concat_panoptic_ids, 0),
            axis=0,
            keepdims=True)
        panoptic = tf.where(
            tf.reduce_any(propagate_mask, axis=0, keepdims=True),
            panoptic_to_replace, next_panoptic)
        panoptic = tf.ensure_shape(panoptic, next_panoptic.get_shape())
        map_concat_to_next.remove(map_concat_to_next.export()[0])
        map_next_to_concat.remove(map_next_to_concat.export()[0])
        return panoptic

    def call(self, concat_panoptic: torch.Tensor,
             next_panoptic: torch.Tensor) -> torch.Tensor:
        """Stitches the prediction from concat_panoptic and next_panoptic.

        Args:
          concat_panoptic: A tf.Tensor for the concat panoptic prediction.
          next_panoptic: A tf.Tensor for the next panoptic prediction.

        Returns:
          A tf.Tensor for the next panoptic prediction with instance ids propagated
            from concat_panoptic based on IoU matching.
        """
        tensorflow_concat_panoptic = tf.convert_to_tensor(concat_panoptic.detach().numpy())
        tensorflow_next_panoptic = tf.convert_to_tensor(next_panoptic.detach().numpy())

        next_panoptic = self._increase_instance_ids_by_offset(
            tensorflow_next_panoptic)  # seperate things and get instance so no stuff got instance id
        concat_panoptic_ids, next_panoptic_ids = (
            self._compute_and_sort_iou_between_panoptic_ids(tensorflow_concat_panoptic,
                                                            next_panoptic))
        panoptic = self._match_and_propagate_instance_ids(tensorflow_concat_panoptic,
                                                          next_panoptic,
                                                          concat_panoptic_ids,
                                                          next_panoptic_ids)
        panoptic = torch.from_numpy(panoptic.numpy())

        return panoptic
