import matplotlib.pyplot as plt
import os
import json

if not os.path.isdir('./graphs'):
    os.mkdir('./graphs')

for log_file in os.scandir('.'):
    if not log_file.name.startswith('log_'):
        continue
    pid = log_file.name[4:-4]
    with open(log_file.name) as f:
        log = f.read()
        log = log.split('\n')
        current_log_data = {'pid': pid, 'epochs': -1, 'losses': [], 'single_losses': []}
        avg_loss_current_epoch = 0
        avg_single_losses_current_epoch = {}
        current_batch_count = 0
        for line in log:
            if line == '':
                continue
            line = json.loads(line)
            if line['type'] == 'epoch_start':
                if current_log_data['epochs'] >= 0 and current_batch_count > 0:
                    current_log_data['losses'].append(avg_loss_current_epoch / current_batch_count)
                    losses_list = {}
                    for loss in avg_single_losses_current_epoch:
                        losses_list[loss] = avg_single_losses_current_epoch[loss] / current_batch_count
                    current_log_data['single_losses'].append(losses_list)
                current_log_data['epochs'] += 1
                avg_loss_current_epoch = 0
                avg_single_losses_current_epoch = {}
                current_batch_count = 0
            if line['type'] == 'sum_loss_output':
                avg_loss_current_epoch += line['value']['loss']
                current_batch_count += 1
            if line['type'] == 'all_losses_output':
                for loss in line['value']:
                    if loss not in avg_single_losses_current_epoch:
                        avg_single_losses_current_epoch[loss] = 0
                    avg_single_losses_current_epoch[loss] += line['value'][loss]
        if current_batch_count > 0:
            current_log_data['losses'].append(avg_loss_current_epoch / current_batch_count)
            losses_list = {}
            for loss in avg_single_losses_current_epoch:
                losses_list[loss] = avg_single_losses_current_epoch[loss] / current_batch_count
            current_log_data['single_losses'].append(losses_list)
        plt.figure()
        plt.plot(current_log_data['losses'])
        plt.savefig('./graphs/' + str(pid) + '_sum_loss.png')
        plt.close()
        losses_x = {}
        for e in range(len(current_log_data['single_losses'])):
            for loss in current_log_data['single_losses'][e]:
                if loss not in losses_x:
                    losses_x[loss] = []
                losses_x[loss].append(current_log_data['single_losses'][e][loss])
        current_log_data['single_losses'] = losses_x
        print(current_log_data)
        for loss in losses_x:
            plt.figure()
            plt.plot(losses_x[loss])
            plt.savefig('./graphs/' + str(pid) + '_' + loss + '_loss.png')
            plt.close()
            
            

