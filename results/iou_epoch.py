import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # path ='Xp/callback_masks'

    # train = pd.read_csv(path + '/unet_1_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/unet_1_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/unet_1_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/multiresunet_1_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/multiresunet_1_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/multiresunet_1_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/multiresunet_5_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/multiresunet_5_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/multiresunet_5_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/nestnet_1_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/nestnet_1_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/nestnet_1_test_proportions.txt', sep="\n", header=None)

    path ='V/callback_masks'

    # train = pd.read_csv(path + '/multiresunet_3_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/multiresunet_3_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/multiresunet_3_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/dunet_3_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/dunet_3_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/dunet_3_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/dunet_5_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/dunet_5_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/dunet_5_test_proportions.txt', sep="\n", header=None)

    # train = pd.read_csv(path + '/nestnet_3_iou_hist.txt', sep="\n", header=None)
    # val = pd.read_csv(path + '/nestnet_3_val_iou_hist.txt', sep="\n", header=None)
    # test = pd.read_csv(path + '/nestnet_3_test_proportions.txt', sep="\n", header=None)

    train = pd.read_csv(path + '/nestnet_5_iou_hist.txt', sep="\n", header=None)
    val = pd.read_csv(path + '/nestnet_5_val_iou_hist.txt', sep="\n", header=None)
    test = pd.read_csv(path + '/nestnet_5_test_proportions.txt', sep="\n", header=None)

    # network = 'U-Net 1 channel'
    # network = 'MultiResU-Net 1 channel'
    # network = 'MultiResU-Net 3 channel'
    # network = 'MultiResU-Net 5 channel'
    # network = 'DU-Net 3 channel'
    # network = 'DU-Net 5 channel'
    # network = 'U-Net++ 1 channel'
    # network = 'U-Net++ 3 channel'
    network = 'U-Net++ 5 channel'

    frames = [train, val, test]
    result = pd.concat(frames, axis=1)
    result.columns = ['train', 'val', 'test']
    epochs = range(0,result['train'].shape[0])
    epochs2 = range(0,result['test'].shape[0])

    sns.set()
    sns.set_context("notebook",font_scale=1.5)
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
    max = np.max(result['val'])
    true_prop = 0.304
    textstrs = r'Max validation $=%.3f$' % (max, )
    textstrs2 = r'Final predicted proportion $=%.3f$' % (result['test'].iloc[-1], )

    axes[0].plot(epochs, result['train'])
    axes[0].plot(epochs, result['val'], ls = '--', color = 'tab:green')
    axes[0].axhline(y=max, xmin=0, xmax=train.shape[0], ls = '-.',label=textstrs, color = 'tab:red')
    axes[0].set(xlabel='Epochs', ylabel='IoU', title ='Training and validation IoU for ' + network)
    axes[0].legend(['Train', 'Validation',textstrs ], ncol=1, loc='best')

    axes[1].plot(epochs2, result['test'])
    axes[1].axhline(y=true_prop, xmin=0, xmax=test.shape[0], ls = '--', color = 'tab:green')
    axes[1].axhline(y=result['test'].iloc[-1], xmin=0, xmax=test.shape[0], ls = '-.', color = 'tab:red')
    axes[1].set(xlabel='Epochs', ylabel='Proportion', title ='Predicted proportions for ' + network)
    axes[1].legend(['Predicted proportion', r'True proportion $=%.3f$' % (true_prop, ), textstrs2], ncol=1, loc='best')

    plt.tight_layout()

    plt.show()
