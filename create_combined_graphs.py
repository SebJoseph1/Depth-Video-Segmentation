import matplotlib.pyplot as plt
import os
graphs = [
    [
        # Data of create_graphs.py and create_eval_graphs.py needs to be inserted here!  
    ],
]
if not os.path.isdir('./finalgraphs'):
    os.mkdir('./finalgraphs')
for g in graphs:
    g[0]['losses'] = g[0]['losses'][0:20]
    g[1]['loss'] = g[1]['loss'][0:20]
    
    
    plt.figure()
    if g[0]['pid'] == 'INSERT PID':
        plt.title('Semkitti dataset, no additional depth losses')
    elif g[0]['pid'] == 'INSERT PID':
        plt.title('Semkitti dataset, with additional depth losses')
    elif g[0]['pid'] == 'INSERT PID':
        plt.title('Cityscapes dataset, with additional depth losses')
    elif g[0]['pid'] == 'INSERT PID':
        plt.title('Cityscapes dataset, no additional depth losses')
    plt.plot(g[0]['losses'], label='training')
    plt.plot(g[1]['loss'], label='validation')
    plt.xticks(range(0, 20))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.xlim()
    plt.savefig('./finalgraphs/' + str(g[0]['pid']) + '.png')
    plt.close()

    for loss in g[0]['single_losses']:
        l_d = g[0]['single_losses'][loss][0:20]
        l_d_v = g[1]['single_losses'][loss][0:20]
        plt.figure()
        if g[0]['pid'] == 'INSERT PID':
            plt.title('Semkitti dataset, no additional depth losses')
        elif g[0]['pid'] == 'INSERT PID':
            plt.title('Semkitti dataset, with additional depth losses')
        elif g[0]['pid'] == 'INSERT PID':
            plt.title('Cityscapes dataset, with additional depth losses')
        elif g[0]['pid'] == 'INSERT PID':
            plt.title('Cityscapes dataset, no additional depth losses')
        plt.plot(l_d, label='training')
        plt.plot(l_d_v, label='validation')
        plt.xticks(range(0, 20))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(loss.replace('_', ' ').title())
        plt.xlim()
        plt.savefig('./finalgraphs/' + str(g[0]['pid']) + '_' + loss + '.png')
        plt.close()