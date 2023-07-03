# ------------------------------------------------------------------------------
# Generates targets for Panoptic-DeepLab.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np
import random
import torch
from dataloader import SemKittiDataset, SemKittiClass
from PIL import Image, ImageOps
from IPython.display import display
import cv2


class PanopticTargetGenerator(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation.
    """

    def __init__(self, thing_list, ignore_label_instance=0, sigma=8,
                 small_instance_area=0, small_instance_weight=1):
        self.ignore_label = ignore_label_instance
        self.thing_list = thing_list
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def create_panoptic_truth(self, semantic, instance):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
                - foreground: Tensor, foreground mask label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(1, H, W).
                - center_points: List, center coordinates, with tuple (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
                - semantic_weights: Tensor, loss weight for semantic prediction, shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction, shape=(H, W), used as weights for center
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction, shape=(H, W), used as weights for offset
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
        """

        height, width = semantic.shape[0], semantic.shape[1]
        panoptic = torch.where(semantic == 255, semantic, semantic * 1000 + instance)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = {}
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(semantic, dtype=np.float32)
        x_coord = np.ones_like(semantic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(semantic, dtype=np.uint8)
        for seg in torch.unique(instance):
            if seg.item() == self.ignore_label:
                continue
            instance_index = torch.nonzero(instance == seg)[0]
            cat_id = semantic[instance_index[0], instance_index[1]]
            panoptic_id = panoptic[instance_index[0], instance_index[1]]
            if cat_id.item() in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic == panoptic_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                # if ins_area < self.small_instance_area:
                semantic_weights[panoptic == panoptic_id] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts[seg.item()] = (center_y, center_x)

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        return dict(
            semantic=semantic.type(torch.long).unsqueeze(0),
            center=torch.as_tensor(center.astype(np.float32)),
            offset=torch.as_tensor(offset.astype(np.float32)),
            center_pts=center_pts,
            semantic_weights=torch.as_tensor(semantic_weights.astype(np.float32)).unsqueeze(0)
        )

    def create_nxt_offset(self, current_panoptic, next_panoptic, current_inst, next_inst, ignore_inst=0):
        nxt_offset = torch.zeros_like(current_panoptic['offset'])
        for inst in torch.unique(next_inst):
            if inst.item() == ignore_inst:
                continue
            if inst in torch.unique(current_inst):
                current_center = current_panoptic['center_pts'][inst.item()]
                next_center = next_panoptic['center_pts'][inst.item()]
                regr_center_y = next_center[0] - current_center[0]
                regr_center_x = next_center[1] - current_center[1]
                mask = next_inst == inst
                nxt_offset[0][mask] = regr_center_y
                nxt_offset[1][mask] = regr_center_x
        return nxt_offset

    def create_vip_deeplab_truth(self, current_image):
        truth_label = {}
        for i in range(len(current_image[0]) - 1):
            current_image_semantic = current_image[1][i].squeeze(0)
            current_image_instance = current_image[2][i].squeeze(0)

            next_image_semantic = current_image[1][i+1].squeeze(0)
            next_image_instance = current_image[2][i+1].squeeze(0)
            current_image_panoptic = self.create_panoptic_truth(current_image_semantic, current_image_instance)
            next_image_panoptic = self.create_panoptic_truth(next_image_semantic, next_image_instance)
            current_image_nxt_offset = self.create_nxt_offset(current_image_panoptic, next_image_panoptic,
                                                            current_image_instance, next_image_instance)
            if i == 0:
                truth_label['semantic'] = current_image_panoptic['semantic'].unsqueeze(0)
                truth_label['center'] = current_image_panoptic['center'].unsqueeze(0)
                truth_label['offset'] = current_image_panoptic['offset'].unsqueeze(0)
                truth_label['nxtoffset'] = current_image_nxt_offset.unsqueeze(0)
                truth_label['semantic_weights'] = current_image_panoptic['semantic_weights'].unsqueeze(0)
            else:
                truth_label['semantic'] = torch.cat([truth_label['semantic'],current_image_panoptic['semantic'].unsqueeze(0)],dim=0)
                truth_label['center'] = torch.cat([truth_label['center'],current_image_panoptic['center'].unsqueeze(0)],dim=0)
                truth_label['offset'] = torch.cat([truth_label['offset'],current_image_panoptic['offset'].unsqueeze(0)],dim=0)
                truth_label['nxtoffset'] = torch.cat([truth_label['nxtoffset'],current_image_nxt_offset.unsqueeze(0)],dim=0)
                truth_label['semantic_weights'] = torch.cat([truth_label['semantic_weights'],current_image_panoptic['semantic_weights'].unsqueeze(0)],dim=0)
        truth_label['depth'] = current_image[0][:-1] / 256.0
        return truth_label


classes = {
    #                 name                     ID       hasInstances    color
    0: SemKittiClass('car', 0, True, (0, 0, 255)),
    1: SemKittiClass('bicycle', 1, True, (245, 150, 100)),
    2: SemKittiClass('motorcycle', 2, True, (245, 230, 100)),
    3: SemKittiClass('truck', 3, True, (250, 80, 100)),
    4: SemKittiClass('other-vehicle', 4, True, (150, 60, 30)),
    5: SemKittiClass('person', 5, True, (111, 74, 0)),
    6: SemKittiClass('bicyclist', 6, True, (81, 0, 81)),
    7: SemKittiClass('motorcyclist', 7, True, (128, 64, 128)),
    8: SemKittiClass('road', 8, False, (244, 35, 232)),
    9: SemKittiClass('parking', 9, False, (250, 170, 160)),
    10: SemKittiClass('sidewalk', 10, False, (230, 150, 140)),
    11: SemKittiClass('other-ground', 11, False, (70, 70, 70)),
    12: SemKittiClass('building', 12, False, (102, 102, 156)),
    13: SemKittiClass('fence', 13, False, (190, 153, 153)),
    14: SemKittiClass('vegetation', 14, False, (180, 165, 180)),
    15: SemKittiClass('trunk', 15, False, (150, 100, 100)),
    16: SemKittiClass('terrain', 16, False, (150, 120, 90)),
    17: SemKittiClass('pole', 17, False, (153, 153, 153)),
    18: SemKittiClass('traffic-sign', 18, False, (50, 120, 255)),
    255: SemKittiClass('unlabeled', 255, False, (0, 0, 0)),
}

if __name__ == "__main__":
    model = PanopticTargetGenerator(thing_list=np.arange(8))
