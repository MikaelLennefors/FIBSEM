import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path ='../results/Xp/weights/multiresunet_1_weights.txt2'
weights = pd.read_csv(path, sep=",")# , header=None


if __name__ == '__main__':
    print(np.max(weights))
    print(np.min(weights))
    # print(weights.iloc[0:10])
    print(weights.shape)
    # sns.set(rc={'figure.facecolor':'gray'})
    # sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_context("notebook",font_scale=1.5)
    fig,axes=plt.subplots(1,1,figsize=(16,9))
    # fig.suptitle('Distribution of weights for MultiResU-net 1 channel', fontsize=18, fontweight="bold")
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    max = np.max(weights)
    min = np.min(weights)
    textstrs = '\n'.join((
        r'max $=%.2f$' % (max, ),
        r'min $=%.2f$' % (min, )))
    sns.distplot(weights, kde=False, bins = 100)
    axes.set(xlabel='Weight size', ylabel='Frequency', title ='Distribution of weights for MultiResU-Net 1 channel')
    axes.text(0.05, 0.95, textstrs, transform=axes.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    axes.set_yscale('log')

    plt.show()
