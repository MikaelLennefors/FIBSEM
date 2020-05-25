import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path ='Xp/callback_masks'

    # train = pd.read_csv(path + '/unet_1_iou_hist.txt', sep=",")
    # val = pd.read_csv(path + '/unet_1_val_iou_hist.txt', sep=",")
    # test = pd.read_csv(path + '/unet_1_test_proportions.txt', sep=",")

    # train = pd.read_csv(path + '/multiresunet_1_iou_hist.txt', sep=",")
    # val = pd.read_csv(path + '/multiresunet_1_val_iou_hist.txt', sep=",")
    # test = pd.read_csv(path + '/multiresunet_1_test_proportions.txt', sep=",")

    # train = pd.read_csv(path + '/multiresunet_5_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/multiresunet_5_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/multiresunet_5_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/nestnet_1_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/nestnet_1_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/nestnet_1_test_proportions.txt', sep="\n", header=None)

    train = pd.read_csv(path + '/nestnet_3_iou_hist.txt', sep=",")
    val = pd.read_csv(path + '/nestnet_3_val_iou_hist.txt', sep=",")
    test = pd.read_csv(path + '/nestnet_3_test_proportions.txt', sep=",")

    # path ='V/callback_masks'

    # train = pd.read_csv(path + '/multiresunet_3_iou_hist.txt', sep=",")
    # val = pd.read_csv(path + '/multiresunet_3_val_iou_hist.txt', sep=",")
    # test = pd.read_csv(path + '/multiresunet_3_test_proportions.txt', sep=",")

    # train = pd.read_csv(path + '/dunet_3_iou_hist.txt', sep=",")
    # val = pd.read_csv(path + '/dunet_3_val_iou_hist.txt', sep=",")
    # test = pd.read_csv(path + '/dunet_3_test_proportions.txt', sep=",")

    # train = pd.read_csv(path + '/dunet_5_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/dunet_5_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/dunet_5_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/nestnet_5_iou_hist.txt', sep=",")
    # val = pd.read_csv(path + '/nestnet_5_val_iou_hist.txt', sep=",")
    # test = pd.read_csv(path + '/nestnet_5_test_proportions.txt', sep=",")

    # network = 'U-Net 1 channel'
    # network = 'MultiResU-Net 1 channel'
    # network = 'MultiResU-Net 3 channel'
    # network = 'MultiResU-Net 5 channel'
    # network = 'DU-Net 3 channel'
    # network = 'DU-Net 5 channel'
    # network = 'U-Net++ 1 channel'
    network = 'U-Net++ 3 channel'
    # network = 'U-Net++ 5 channel'
    epochs = range(0,train.shape[0])
    epochs_test = range(0,test.shape[0])
    train['epoch'] = epochs
    # IoU_train = pd.concat([train['2'], train['3']],axis = 0)
    # epoch = pd.concat([train['epoch'], train['epoch']])
    IoU_train = pd.concat([train['1'],train['2'], train['3']],axis = 0)
    epoch = pd.concat([train['epoch'],train['epoch'], train['epoch']])
    train = pd.concat([IoU_train, epoch],axis = 1)
    train.columns = ['IoU_train', 'epoch']


    # val= pd.concat([val['2'], val['3']],axis = 1)
    mean_val = np.mean(val,axis =1)
    max_val = np.max(mean_val)
    val['epoch'] = epochs
    # IoU_val = pd.concat([val['2'], val['3']],axis = 0)
    # epoch = pd.concat([val['epoch'], val['epoch']])
    IoU_val = pd.concat([val['1'],val['2'], val['3']],axis = 0)
    epoch = pd.concat([val['epoch'],val['epoch'], val['epoch']])
    val = pd.concat([IoU_val, epoch],axis = 1)
    val.columns = ['IoU_val', 'epoch']
    print(test)
    # test = pd.concat([test['2'], test['3']],axis = 1)
    mean_prop = np.mean(test.iloc[-1])
    std_prop = np.std(test.iloc[-1])
    test['epoch'] = epochs_test
    prop = pd.concat([test['2'], test['3']],axis = 0)
    epoch = pd.concat([test['epoch'], test['epoch']])
    # prop = pd.concat([test['1'],test['2'], test['3']],axis = 0)
    # epoch = pd.concat([test['epoch'],test['epoch'], test['epoch']])
    test = pd.concat([prop, epoch],axis = 1)
    test.columns = ['Prop', 'epoch']

    sns.set_context("notebook",font_scale=1.5)
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
    true_prop = 0.304

    textstrs = r'Max validation $=%.3f$' % (max_val, )
    # textstrs = 'Max validation {:3f}({:3f})'.format(mean_val,std_val)
    # textstrs = ''.join((
    #     r'Max validation $=%.3f$' % (mean_val, ),
    #     r'$\pm%.3f$' % (std_val, )))
    # print(textstrs)

    # textstrs2 = r'Final predicted proportion $=%.3f$' % (result['test'].iloc[-1], )
    textstrs2 = ''.join((
        r'Final predicted proportion $=%.3f$' % (mean_prop, ),
        r'$\pm%.3f$' % (std_prop, )))

    # axes[0].plot(epochs, result['train'])
    # axes[0].plot(epochs, result['val'], ls = '--', color = 'tab:green')

    sns.lineplot(x = 'epoch', y = 'IoU_train', data = train, ax=axes[0])
    sns.lineplot(x = 'epoch', y = 'IoU_val', data = val, ax=axes[0], color = 'tab:green')

    axes[0].axhline(y=max_val, xmin=0, xmax=train.shape[0]/3, ls = '-.',label=textstrs, color = 'tab:red')
    axes[0].set(xlabel='Epochs', ylabel='IoU', title ='Training and validation IoU for ' + network)
    axes[0].legend(['Train', 'Validation',textstrs ], ncol=1, loc='best')

    # # axes[1].plot(epochs2, result['test'])
    sns.lineplot(x = 'epoch', y = 'Prop', data = test, ax=axes[1])
    axes[1].axhline(y=true_prop, xmin=0, xmax=test.shape[0], ls = '--', color = 'tab:green')
    axes[1].axhline(y=mean_prop, xmin=0, xmax=test.shape[0], ls = '-.', color = 'tab:red')
    axes[1].set(xlabel='Epochs', ylabel='Proportion', title ='Predicted proportions for ' + network)
    axes[1].legend(['Predicted proportion', r'True proportion $=%.3f$' % (true_prop, ), textstrs2], ncol=1, loc='best')

    plt.tight_layout()

    plt.show()
