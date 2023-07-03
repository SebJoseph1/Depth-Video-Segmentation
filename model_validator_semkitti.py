import sys
pid = sys.argv[1]
import os
from importlib.machinery import SourceFileLoader
import torch

USE_ADDITIONAL_DEPTH_LOSS = 'use_depth_loss'in sys.argv

if USE_ADDITIONAL_DEPTH_LOSS:
    print('Using additional depth loss')

import dataloader
from dataloader import SemKittiDataset
model_processing = SourceFileLoader("model_processing","./model+processing.py").load_module()
from model_processing import PanopticModel
from truthprocess import PanopticTargetGenerator
from utils import log_to_json
from validatemodel import ValidationDepthLoss

if USE_ADDITIONAL_DEPTH_LOSS:
    from losstorch_new_depth import sum_vip_deeplab_losses
else:
    from losstorch_new import sum_vip_deeplab_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
log_to_json('I', 'Device used ' + str(device))

validator = ValidationDepthLoss(ignoreLabel=0.0)

for model_file in os.scandir('.'):
    model_file = model_file.name
    if not model_file.startswith('model_' + str(pid)):
        continue
    print('Starting testing')
    log_to_json('I', 'Starting Testing. Additional loss: '  + str(USE_ADDITIONAL_DEPTH_LOSS))
    val_data_loader = SemKittiDataset(dir_input="C:/Users/Sebastian Joseph/Downloads/semkitti-dvps/semkitti-dvps/video_sequence/val", classes=dataloader.classes, max_batch_size=2, shuffle=True)
    #training_data_loader = torch.utils.data.DataLoader(training_loader, batch_size=3, shuffle=False)
    print('Dataset created. Len: ' + str(len(val_data_loader)))
    log_to_json('I', 'Dataset created')
    model_list = []
    model = PanopticModel(weight=model_file)
    model.to(device)
    print('Model created')

    log_to_json('I', 'Training loop started')
    log_to_json('model_name', model_file)

    converter = PanopticTargetGenerator(dataloader.thing_list, small_instance_area=10, small_instance_weight=3)
    
    total_loss = 0

    model.model.eval()
    for i in range(len(val_data_loader)):
        data = val_data_loader[i]
        inputs, labels = data
        labels = converter.create_vip_deeplab_truth(labels)
        if len(inputs) < 2:
            continue
        inputs = inputs[:-1]
        log_to_json('processing_batch', {'id': i, 'length': len(inputs)})
        with torch.no_grad():
            inputs.requires_grad = False
            inputs = inputs.to(device)
            for key in labels:
                labels[key].requires_grad = False
                labels[key] = labels[key].to(device)
            outputs = model(inputs)
            

            if USE_ADDITIONAL_DEPTH_LOSS:
                loss = sum_vip_deeplab_losses(inputs, outputs, labels)
            else:
                loss = sum_vip_deeplab_losses(outputs, labels)
            log_to_json('sum_loss_output', {'loss': loss.item()})

            avg_loss = loss.item()
            total_loss += avg_loss
            validator_losses = validator(labels['depth'], outputs['depth'])
            for v in validator_losses:
                validator_losses[v] = validator_losses[v].item()
            log_to_json('validator_output', validator_losses)
    print("AVG total loss {}".format(total_loss / 20))
    log_to_json('val_loss', total_loss / 20)