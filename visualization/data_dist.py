import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(1, '../processing')
from extract_data import extract_data

data_path = '../data/train_val_data_border_clean/'
test_path = '../data/test_data_border_clean/'

sns.set(color_codes=True)

def vis_data_dist(channels = 7):
    images, masks = extract_data(data_path, channels)
    test_img, test_mask = extract_data(test_path, channels)

    img_train = images[0,:,:,0]
    mask_train = masks[0,:,:,0]
    indices_train = np.nonzero(mask_train)
    np.add.at(indices_train[0], range(0,len(indices_train[0])), 1)
    np.add.at(indices_train[1], range(0,len(indices_train[0])), 1)
    por_train = img_train[indices_train]
    bakgrund_train = np.delete(img_train,indices_train)

    mu_por_train = por_train.mean()
    median_por_train = np.median(por_train)
    sigma_por_train = por_train.std()
    textstr_por_train = '\n'.join((
        r'$\mu=%.2f$' % (mu_por_train, ),
        r'$\mathrm{median}=%.2f$' % (median_por_train, ),
        r'$\sigma=%.2f$' % (sigma_por_train, )))

    mu_bak_train = bakgrund_train.mean()
    median_bak_train = np.median(bakgrund_train)
    sigma_bak_train = bakgrund_train.std()
    textstr_bak_train = '\n'.join((
        r'$\mu=%.2f$' % (mu_bak_train, ),
        r'$\mathrm{median}=%.2f$' % (median_bak_train, ),
        r'$\sigma=%.2f$' % (sigma_bak_train, )))

    fig,axes=plt.subplots(1,2,figsize=(16,9))
    sns.distplot(por_train, kde=False, fit=stats.norm, ax=axes[0])
    axes[0].set(xlabel='Intensity', ylabel='Density', title = 'Distribution of pores')
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    axes[0].text(0.05, 0.95, textstr_por_train, transform=axes[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    sns.distplot(bakgrund_train, kde=False, fit=stats.norm, ax=axes[1], color="r")
    axes[1].set(xlabel='Intensity', ylabel='Density', title = 'Distribution of background')
    axes[1].text(0.05, 0.95, textstr_bak_train, transform=axes[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    fig.suptitle('Training set\n', fontsize=18, fontweight="bold")
    # fig.suptitle('Test set\n', fontsize=18, fontweight="bold")
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    vis_data_dist()
