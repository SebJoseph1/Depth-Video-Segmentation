import sys
USE_ADDITIONAL_DEPTH_LOSS = 'use_depth_loss'in sys.argv

USE_NEW_DEPTH_LOSS = 'use_new_depth_loss'in sys.argv

if USE_ADDITIONAL_DEPTH_LOSS:
    print('Using additional depth loss')
if USE_NEW_DEPTH_LOSS:
    print('Using new depth loss')

import os
print(os.getpid())
import collections
import random
import time
from typing import Any, Dict, Text, Tuple

# import tensorflow as tf
import torch
import torch.nn as nn

import psutil

import dataloader
from dataloader import SemKittiDataset
if USE_ADDITIONAL_DEPTH_LOSS and USE_NEW_DEPTH_LOSS:
    from losstorch_new_depth import sum_vip_deeplab_losses
elif USE_ADDITIONAL_DEPTH_LOSS:
    from losstorch_depth import sum_vip_deeplab_losses
elif USE_NEW_DEPTH_LOSS:
    from losstorch_new import sum_vip_deeplab_losses
else:
    from losstorch import sum_vip_deeplab_losses
from truthprocess_depth import PanopticTargetGenerator
from post_process import PostProcessor
from samplemodel import MonoDVPS
from video_preposses import VideoPanopticPredictionStitcher

from utils import log_to_json

#torch.autograd.set_detect_anomaly(True)  # (Kurt) enable more specific errors in automatic differentiation at the cost of decreased speed. Remove when the issue has been fixed.


class PanopticModel(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super().__init__(**kwargs)
        self.model = MonoDVPS(num_classes=len(dataloader.classes))
        if weight is not None:
            self.model.load_state_dict(torch.load(weight))
        self.video_stitcher = VideoPanopticPredictionStitcher(label_divisor=1000)
        self.post_processor = PostProcessor()

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

                # concat_result_batch_post_process = {}
                # for key,value in self.post_processor.call(concat_result_batch).items():
                #     concat_result_batch_post_process[key] = value
                concat_result_batch_post_process = self.post_processor.call(concat_result_batch)
                next_result_batch_post_process = self.post_processor.call(next_result_batch)

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
                    concat_result[result_key], split_size_or_sections=int(concat_result[result_key].shape[2] / 2),
                    dim=2)
            result['concat_next_panoptic'] = next_result[
                'panoptic']
            result['next_panoptic'] = self.video_stitcher.call(
                result['concat_next_panoptic'],
                result['next_panoptic'])
            result['next_panoptic'].reshape(
                result['concat_next_panoptic'].shape)
        if 'center' in result:
            result['center'] = torch.squeeze(
                result['center'], dim=1)
        #result = depth_post_process(result)
        return result



def train():
    print('Starting training')
    log_to_json('I', 'Starting Training. Additional loss: '  + str(USE_ADDITIONAL_DEPTH_LOSS) + '. New loss: ' + str(USE_NEW_DEPTH_LOSS))
    training_data_loader = SemKittiDataset(dir_input="semkitti-dvps/video_sequence/train", classes=dataloader.classes, max_batch_size=8, shuffle=True)
    #training_data_loader = torch.utils.data.DataLoader(training_loader, batch_size=3, shuffle=False)
    print('Dataset created')
    log_to_json('I', 'Dataset created')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    log_to_json('I', 'Device used ' + str(device))
    model = PanopticModel()
    model.to(device)
    print('Model created')
    EPOCHS = 20
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.00025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    epoch_number = 0


    best_vloss = 1000000.0

    log_to_json('I', 'Training loop started')

    converter = PanopticTargetGenerator(dataloader.classes, small_instance_area=10, small_instance_weight=3)

    USE_CACHE = False

    if USE_CACHE:
        cache = {}
        print('Cache length will be ' + str(len(training_data_loader)))
        for i in range(len(training_data_loader)):
            if i > 10000:
                continue
            if i % 10 == 0:
                print(psutil.virtual_memory())
            print(str(i) + ' ', end='')
            cache[i] = training_data_loader[i]
            cache[i][1] = converter.create_vip_deeplab_truth(cache[i][1])

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        log_to_json('epoch_start', epoch_number)

        model.train()

        print(len(training_data_loader))
        for i in range(len(training_data_loader)):
            print(i)
            if USE_CACHE and i in cache:
                cache_inputs, cache_labels = cache[i]
                inputs = cache_inputs.clone()
                labels = {}
                for key in cache_labels:
                    labels[key] = cache_labels[key].clone()
            else:
                data = training_data_loader[i]
                inputs, labels = data
                labels = converter.create_vip_deeplab_truth(labels)
            log_to_json('processing_batch', {'id': i, 'length': len(inputs)})
            if len(inputs) < 3:
                continue
            inputs = inputs.to(device)
            for key in labels:
                labels[key] = labels[key].to(device)


            optimizer.zero_grad()
            inputs = inputs[:-1]
            outputs = model(inputs)
            
            if USE_ADDITIONAL_DEPTH_LOSS:
                loss = sum_vip_deeplab_losses(inputs, outputs, labels)
            else:
                loss = sum_vip_deeplab_losses(outputs, labels)
            log_to_json('sum_loss_output', {'epoch': epoch_number, 'batch': i, 'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]["lr"]})
            loss.backward()

            optimizer.step()

            print("mean weight",torch.mean(model.model.semantic_pred_head.first_layer[0][0].weight).detach().item())

            avg_loss = loss.item()
            batch_size = len(inputs)

            print('LOSS train {}'.format(avg_loss))
        scheduler.step()

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_path = 'model_{}_{}_{}'.format(os.getpid(), timestamp, epoch_number)
        torch.save(model.model.state_dict(), model_path)

        epoch_number += 1


# Depth head
def depth_post_process(results, min_depth=0, max_depth=88):  # not sure if needed
    depth_prediction = results['depth']
    pred_value = min_depth + torch.sigmoid(depth_prediction) * (
            max_depth - min_depth)
    results['depth'] = pred_value
    return results


train()

# z = PanopticModel()
# k = torch.randn(1, 6, 300, 300)
# j = z.predict(input_tensor=k)
# print(j)
# we should implement the inference in the vip-deeplab-demo
