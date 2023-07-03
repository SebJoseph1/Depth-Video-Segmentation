import collections
import cv2
import time
from typing import Any, Dict, Text, Tuple
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataloader
from dataloader import SemKittiDataset
from losstorch import (L1Loss, MSELoss, SILogPlusRelativeSquaredLoss,
                       TopKCrossEntropyLoss, sum_vip_deeplab_losses)
from post_process import PostProcessor
from samplemodel import MonoDVPS
from truthprocess import PanopticTargetGenerator
from video_preposses import VideoPanopticPredictionStitcher
import numpy as np
from validatemodel import ValidationDepthLoss

torch.autograd.set_detect_anomaly(True)  # (Kurt) enable more specific errors in automatic differentiation at the cost of decreased speed. Remove when the issue has been fixed.


class PanopticModel(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super().__init__(**kwargs)
        self.model = MonoDVPS(num_classes=len(dataloader.classes)-1)
        if weight is not None:
            self.model.load_state_dict(torch.load(weight))
        self.video_stitcher = VideoPanopticPredictionStitcher(label_divisor=1000)
        self.post_processor = PostProcessor()
        self.semantic_loss = TopKCrossEntropyLoss(num_classes=19, ignore_label=32)
        self.center_loss = MSELoss(weight=200)
        self.regression_loss = L1Loss(weight=0.01)
        self.next_regression_loss = L1Loss(weight=0.01)
        self.depth_loss = SILogPlusRelativeSquaredLoss(ignore_label=0, weight=0.1)
        self.truth_post_process = PanopticTargetGenerator(thing_list=list(range(8)), small_instance_area=10,
                                                          small_instance_weight=3)

    def forward(self,
                input_tensor: torch.Tensor,
                ) -> Dict[Text, Any]:

        input_tensor = input_tensor / 127.5 - 1.0
        if self.training:
            result = self.model(input_tensor)
        else:
            result = self.model(input_tensor)
            input_swapped = input_tensor.clone()
            input_swapped[:, :3, :, :], input_swapped[:, 3:, :, :] = input_tensor[:, 3:, :, :], input_tensor[:, :3, :,
                                                                                                :]
            next_result = self.model.forward(input_swapped)

            concat_result = collections.defaultdict(list)
            concat_result['semantic'] = torch.concat([
                result['semantic'],
                next_result['semantic']], dim=3)  # concat width wise
            concat_result['center'] = torch.concat([
                result['center'],
                torch.zeros_like(next_result['center'])], dim=3)
            next_regression_y, next_regression_x = torch.split(
                result['nxtoffset'],
                split_size_or_sections=1,
                dim=1)  # divide
            next_regression_x = next_regression_x - input_tensor.shape[3]
            next_regression = torch.concat([next_regression_y, next_regression_x], dim=1)
            concat_result['offset'] = torch.concat(
                [result['offset'], next_regression], dim=3)
            batch_size = input_tensor.shape[0]
            concat_result_combined = {}
            next_result_combined = {}
            for batch_index in range(batch_size):
                concat_result_batch = collections.defaultdict(list)
                next_result_batch = collections.defaultdict(list)
                concat_result_batch['semantic'] = concat_result['semantic'][batch_index].unsqueeze(0)
                concat_result_batch['center'] = concat_result['center'][batch_index].unsqueeze(0)
                concat_result_batch['offset'] = concat_result['offset'][batch_index].unsqueeze(0)
                next_result_batch['semantic'] = next_result['semantic'][batch_index].unsqueeze(0)
                next_result_batch['center'] = next_result['center'][batch_index].unsqueeze(0)
                next_result_batch['offset'] = next_result['offset'][batch_index].unsqueeze(0)
                concat_result_batch_post_process = self.post_processor.call(concat_result_batch)
                next_result_batch_post_process = self.post_processor.call(next_result_batch)
                for key_value in concat_result_batch_post_process:
                    concat_result_batch_post_process[key_value] = concat_result_batch_post_process[key_value].unsqueeze(0)
                    next_result_batch_post_process[key_value] = next_result_batch_post_process[key_value].unsqueeze(0)
                if batch_index == 0:
                    concat_result_combined = concat_result_batch_post_process
                    next_result_combined = next_result_batch_post_process
                else:
                    for key in concat_result_batch_post_process.keys():
                        concat_result_combined[key] = torch.cat(
                            (concat_result_combined[key], concat_result_batch_post_process[key]), dim=0)
                        next_result_combined[key] = torch.cat(
                            (next_result_combined[key], next_result_batch_post_process[key]), dim=0)

            concat_result.update(concat_result_combined)
            next_result.update(next_result_combined)
            result['next_panoptic'] = next_result[
                'panoptic']

            for result_key in [
                'panoptic', 'semantic',
                'instance', 'center',
                'instance_scores'
            ]:
                result[result_key], next_result[result_key] = torch.split(
                    concat_result[result_key], split_size_or_sections=int(concat_result[result_key].shape[3] / 2),
                    dim=3)
            
            result['concat_next_panoptic'] = next_result[
                'panoptic']
            video_stitcher = {}
            for video_batch_index in range(batch_size):
                if video_batch_index == 0:
                    result_video_stitcher = self.video_stitcher.call(
                        result['concat_next_panoptic'][video_batch_index],
                        result['next_panoptic'][video_batch_index])
                    video_stitcher['next_panoptic'] = result_video_stitcher.unsqueeze(0)
                else:
                    result_video_stitcher = self.video_stitcher.call(
                        result['concat_next_panoptic'][video_batch_index],
                        result['next_panoptic'][video_batch_index])
                    video_stitcher['next_panoptic'] = torch.cat(
                            (video_stitcher['next_panoptic'], result_video_stitcher.unsqueeze(0)), dim=0)
            result['next_panoptic'] = video_stitcher['next_panoptic']
        return result



def evaluate():
    print('Starting evaluation')
    training_data_loader = SemKittiDataset(dir_input="C:/Users/Sebastian Joseph/Downloads/semkitti-dvps/semkitti-dvps/video_sequence/train", classes=dataloader.classes, max_batch_size=2, shuffle=True)
    print('Dataset created')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    model = PanopticModel(weight="C:/Users/Sebastian Joseph/Desktop/Q4/advance sensing/model_3901401_20230624_103500_9")
    model.to(device)
    print('Model created')
    model.eval()
    model.requires_grad_(False)
    converter = PanopticTargetGenerator(thing_list= dataloader.thing_list, small_instance_area=20, small_instance_weight=3)
    print(len(training_data_loader))
    for i in range(len(training_data_loader)):
        data = training_data_loader[i]
        print(i)
        # if i > 3:
        #     break
        if i == len(training_data_loader):
           continue
        inputs, labels = data
        if len(inputs) <= 1:
            continue
        inputs = inputs.to(device)
        # for key in range(len(labels)):
        #     labels[key] = labels[key][:, :, :360, :360]
        # inputs = inputs[:, :, :360, :360]
        labels = converter.create_vip_deeplab_truth(labels)
        for key in labels:
            labels[key] = labels[key].to(device)
        
        inputs = inputs[:-1]
        outputs = model(inputs)
        for outputs_key in outputs:
            outputs[outputs_key] = outputs[outputs_key].to(device)
            if outputs_key == "depth":
                for ij in range(len(outputs[outputs_key])):
                    print("Depth values ",ij,"  ",torch.unique(outputs[outputs_key][ij]))   

        print("mean weight",torch.mean(model.model.semantic_pred_head.first_layer[0][0].weight).detach().item())

        label_divisor = 1000
        overlap_offset = label_divisor // 2
        combine_offset = 2 ** 32
        max_instance_id = 0
        stitched_panoptic = None
        last_panoptic = None
        current_batch_size = outputs["panoptic"].shape[0]
        for current_batch_index in range(current_batch_size - 1):
            current_next_panoptic = outputs["next_panoptic"][current_batch_index]
            current_panoptic = outputs["panoptic"][current_batch_index]
            next_new_mask = current_next_panoptic % label_divisor > overlap_offset #find index of all pixel with instance with 500*
            if last_panoptic is not None:
                intersection = (
                        last_panoptic.type(torch.int64) * combine_offset +
                        current_panoptic.type(torch.int64))
                intersection_ids, intersection_counts = torch.unique(
                        intersection, return_counts=True)
                intersection_ids = intersection_ids[torch.argsort(intersection_counts)]
                for intersection_id in intersection_ids:
                    last_panoptic_id = intersection_id // combine_offset
                    panoptic_id = intersection_id % combine_offset
                    current_next_panoptic[current_next_panoptic == panoptic_id] = last_panoptic_id.type(torch.int32)

            # Adjust the IDs for the new instances in next_panoptic.
            max_instance_id = max(max_instance_id,
                                            torch.max(current_panoptic % label_divisor))# find max value of instance id in panoptic mask
            next_panoptic_cls = current_next_panoptic // label_divisor # find segmentation class of all pixels for next panoptic
            next_panoptic_ins = current_next_panoptic % label_divisor# find instance of all pixels for next panoptic
            next_panoptic_ins[next_new_mask] = (
                    next_panoptic_ins[next_new_mask] - overlap_offset
                    + max_instance_id) # change the instance of next_panoptic which is 500 * to inst + max value
            current_next_panoptic = (
                    next_panoptic_cls * label_divisor + next_panoptic_ins) # create panoptic by combining with class and inst
            if stitched_panoptic is None:
                stitched_panoptic = current_panoptic.unsqueeze(0)
            stitched_panoptic = torch.cat(
                            (stitched_panoptic, current_next_panoptic.unsqueeze(0)), dim=0)
            max_instance_id = max(max_instance_id,
                                            torch.max(current_next_panoptic % label_divisor)) # update to max in next panoptic
            last_panoptic = copy.deepcopy(current_next_panoptic)
        outputs["stitched_panoptic"] = stitched_panoptic

        for batch_idx in range(current_batch_size):
            # Displaying the model prediction
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].title.set_text('Input Image')
            input_image = inputs[batch_idx][:3].permute(1,2,0)
            ax[0].imshow(input_image.detach().cpu())
            ax[1].title.set_text('Depth')
            ax[1].imshow(outputs["depth"][batch_idx][0].cpu().detach().numpy())
            ax[2].title.set_text('Video Panoptic Segmentation')
            panoptic_map = training_data_loader.to_image([outputs['depth'][batch_idx],outputs['semantic'][batch_idx],outputs['depth'][batch_idx]])
            print("Dispalying segmentation")
            ax[2].imshow(panoptic_map[1])

            # Displaying the truth label
            fig1, ax1 = plt.subplots(1, 3, figsize=(18, 6))
            ax1[0].title.set_text('Input Image')
            input_image = inputs[batch_idx][:3].permute(1,2,0)
            ax1[0].imshow(input_image.detach().cpu())
            ax1[1].title.set_text('Depth')
            ax1[1].imshow(labels["depth"][batch_idx][0].cpu().detach().numpy())
            ax1[2].title.set_text('Video Panoptic Segmentation')
            panoptic_map = training_data_loader.to_image([labels['depth'][batch_idx],labels['semantic'][batch_idx],labels['semantic'][batch_idx]])
            ax1[2].imshow(panoptic_map[1])
        
        vloss = ValidationDepthLoss(ignoreLabel=0.0).forward(y_true=labels['depth'],y_pred=outputs['depth'])
        print('LOSS valid absolute relative error {} squared relative error {} root mean squared error {}'.format(vloss["absRelErr"], vloss["squRelErr"], vloss["rootMeanSquErr"]))



# Depth head
def depth_post_process(results, min_depth=0, max_depth=88):  # not sure if needed
    depth_prediction = results['depth']
    pred_value = min_depth + torch.sigmoid(depth_prediction) * (
            max_depth - min_depth)
    results['depth'] = pred_value
    return results


def _cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  See more about CITYSCAPES dataset at https://www.cityscapes-dataset.com/
  M. Cordts, et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR. 2016.

  Returns:
    A 2-D numpy array with each row being mapped RGB color (in uint8 range).
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  colormap[19] = [0, 0, 0]
  return colormap

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    'num_classes, label_divisor, thing_list, colormap, class_names')

def _cityscapes_class_names():
  return ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
          'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',#10
          'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
          'traffic-sign')


def cityscapes_dataset_information():
  return DatasetInfo(
      num_classes=19,
      label_divisor=1000,
      thing_list=tuple(range(0, 8)),
      colormap=_cityscapes_label_colormap(),
      class_names=_cityscapes_class_names())


def perturb_color(color, noise, used_colors, max_trials=50, random_state=None):
  """Pertrubs the color with some noise.

  If `used_colors` is not None, we will return the color that has
  not appeared before in it.

  Args:
    color: A numpy array with three elements [R, G, B].
    noise: Integer, specifying the amount of perturbing noise (in uint8 range).
    used_colors: A set, used to keep track of used colors.
    max_trials: An integer, maximum trials to generate random color.
    random_state: An optional np.random.RandomState. If passed, will be used to
      generate random numbers.

  Returns:
    A perturbed color that has not appeared in used_colors.
  """
  if random_state is None:
    random_state = np.random

  for _ in range(max_trials):
    random_color = color + random_state.randint(
        low=-noise, high=noise + 1, size=3)
    random_color = np.clip(random_color, 0, 255)

    if tuple(random_color) not in used_colors:
      used_colors.add(tuple(random_color))
      return random_color

  print('Max trial reached and duplicate color will be used. Please consider '
        'increase noise in `perturb_color()`.')
  return random_color

def color_panoptic_map(panoptic_prediction,
                       dataset_info,
                       perturb_noise,
                       used_colors,
                       color_mapping):
  """Helper method to colorize output panoptic map.

  Args:
    panoptic_prediction: A 2D numpy array, panoptic prediction from deeplab
      model.
    dataset_info: A DatasetInfo object, dataset associated to the model.
    perturb_noise: Integer, the amount of noise (in uint8 range) added to each
      instance of the same semantic class.
    used_colors: A set, used to keep track of used colors.
    color_mapping: A dict, used to map exisiting panoptic ids.

  Returns:
    colored_panoptic_map: A 3D numpy array with last dimension of 3, colored
      panoptic prediction map.
    used_colors: A dictionary mapping semantic_ids to a set of colors used
      in `colored_panoptic_map`.
  """
  if panoptic_prediction.ndim != 2:
    raise ValueError('Expect 2-D panoptic prediction. Got {}'.format(
        panoptic_prediction.shape))

  semantic_map = panoptic_prediction // dataset_info.label_divisor
  instance_map = panoptic_prediction % dataset_info.label_divisor
  height, width = panoptic_prediction.shape
  colored_panoptic_map = np.zeros((height, width, 3), dtype=np.uint8)

  # Use a fixed seed to reproduce the same visualization.
  random_state = np.random.RandomState(0)

  unique_semantic_ids = np.unique(semantic_map)
  for semantic_id in unique_semantic_ids:
    semantic_mask = semantic_map == semantic_id
    if semantic_id in dataset_info.thing_list:
      # For `thing` class, we will add a small amount of random noise to its
      # correspondingly predefined semantic segmentation colormap.
      unique_instance_ids = np.unique(instance_map[semantic_mask])
      for instance_id in unique_instance_ids:
        instance_mask = np.logical_and(semantic_mask,
                                       instance_map == instance_id)
        panoptic_id = semantic_id * dataset_info.label_divisor + instance_id
        if panoptic_id not in color_mapping:
          random_color = perturb_color(
              dataset_info.colormap[semantic_id],
              perturb_noise,
              used_colors[semantic_id],
              random_state=random_state)
          colored_panoptic_map[instance_mask] = random_color
          color_mapping[panoptic_id] = random_color
        else:
          colored_panoptic_map[instance_mask] = color_mapping[panoptic_id]
    else:
      # For `stuff` class, we use the defined semantic color.
      colored_panoptic_map[semantic_mask] = dataset_info.colormap[semantic_id]
      used_colors[semantic_id].add(tuple(dataset_info.colormap[semantic_id]))
  return colored_panoptic_map

evaluate()

# z = PanopticModel()
# k = torch.randn(1, 6, 300, 300)
# j = z.predict(input_tensor=k)
# print(j)
# we should implement the inference in the vip-deeplab-demo
