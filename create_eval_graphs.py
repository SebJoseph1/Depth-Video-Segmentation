import os
import json

for file in os.scandir('.'):
    if not file.name.startswith('log_'):
        continue
    with open(file.name) as f:
        content = f.read().split('\n')
    current_epoch = -1
    current_data = {'name': file.name, 'loss':[], 'single_losses':{}}
    batches_count = 0
    current_batch_total_loss = 0
    single_losses = {}
    for l in content:
        if l == '':
            continue
        l = json.loads(l)
        if l['type'] == 'model_name':
            if current_epoch > -1:
                current_data['loss'].append(current_batch_total_loss / batches_count)
                for loss in single_losses:
                    if loss not in current_data['single_losses']:
                        current_data['single_losses'][loss] = []
                    current_data['single_losses'][loss].append(single_losses[loss] / batches_count)
            current_epoch = int(l['value'].split('_')[-1])
            current_batch_total_loss = 0
            batches_count = 0
            single_losses = {}
        if l['type'] == 'processing_batch':
            batches_count += 1
        if l['type'] == 'sum_loss_output':
            current_batch_total_loss += l['value']['loss']
        if l['type'] == 'all_losses_output':
            for loss in l['value']:
                if loss not in single_losses:
                    single_losses[loss] = 0
                single_losses[loss] += l['value'][loss]
    print(current_data)