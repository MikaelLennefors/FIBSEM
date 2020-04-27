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

def vis_data_dist(channels = 1):
    images, masks = extract_data(data_path, channels)
    test_img, test_mask = extract_data(test_path, channels)
    print(np.shape(images))
    print(np.shape(masks))

    # img_train = images[:,:,:,:]
    # mask_train = masks[:,:,:,:]
    # indices = np.zeros

    for pic in range(masks.shape[0]):
        indices = np.nonzero(masks[pic,:,:,0])
        np.add.at(indices[0], range(0,len(indices[0])), 1)
        np.add.at(indices[1], range(0,len(indices[1])), 1)
        print(np.shape(indices))
        # indices = np.nonzero(mask_train)
        # np.add.at(indices_train[0], range(0,len(indices_train[0])), 1)
        # np.add.at(indices_train[1], range(0,len(indices_train[0])), 1)
        por = images[pic,indices]
        bakgrund = np.delete(images[pic],indices)

    raise
    mu_por = por.mean()
    median_por = np.median(por)
    sigma_por = por.std()
    textstr_por = '\n'.join((
        r'$\mu=%.2f$' % (mu_por, ),
        r'$\mathrm{median}=%.2f$' % (median_por, ),
        r'$\sigma=%.2f$' % (sigma_por, )))

    mu_bak = bakgrund_train.mean()
    median_bak = np.median(bakgrund)
    sigma_bak_ = bakgrund.std()
    textstr_bak_ = '\n'.join((
        r'$\mu=%.2f$' % (mu_bak, ),
        r'$\mathrm{median}=%.2f$' % (median_bak, ),
        r'$\sigma=%.2f$' % (sigma_bak, )))
    raise
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
