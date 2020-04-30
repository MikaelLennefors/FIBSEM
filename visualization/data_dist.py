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
    test_img, test_masks = extract_data(test_path, channels)

    indices = np.nonzero(masks.ravel())
    print(np.shape(masks))
    print(np.shape(indices))
    print(indices)
    sys.exit()

    images30, _ = extract_data_ratio(data_path, channels, 30)
    test_img30, _ = extract_data_ratio(test_path, channels, 30)
    images30 =  np.hstack([images30.ravel(), test_img30.ravel()])

    images45, _ = extract_data_ratio(data_path, channels, 45)
    test_img45, _ = extract_data_ratio(test_path, channels, 45)
    images45 =  np.hstack([images45.ravel(), images45.ravel()])

    means = np.zeros(3)
    stds = np.zeros(3)
    textstrs = [' ', ' ', ' ']
    name = ['22% pores', '30% pores', '45% pores']
    col = ['b', 'r', 'g']
    ratios = [images22, images30, images45]
    support = np.linspace(0, 255, 1000)

    fig,axes=plt.subplots(1,3,figsize=(24,13.5))
    fig.suptitle('Distribution of intensities', fontsize=18, fontweight="bold")
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

    for i in range(len(ratios)):
        means[i] = ratios[i].mean()
        stds[i] = ratios[i].std()
        textstrs[i] = '\n'.join((
            r'$\mu=%.2f$' % (means[i], ),
            r'$\sigma=%.2f$' % (stds[i], )))
        sns.distplot(ratios[i], kde=True, ax=axes[i], color=col[i])
        axes[i].set(xlabel='Intensity', ylabel='Density', title = name[i])
        axes[i].text(0.05, 0.95, textstrs[i], transform=axes[i].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    # ratio = 30
    # images, masks = extract_data_ratio(data_path, channels, ratio)
    # test_img, test_mask = extract_data_ratio(test_path, channels, ratio)
    # print(np.shape(images.ravel()))
    # # print(np.shape(masks))
    # #
    # # # img_train = images[:,:,:,:]
    # # # mask_train = masks[:,:,:,:]
    # # # indices = np.zeros
    # #
    # # for pic in range(masks.shape[0]):
    # #     indices = np.nonzero(masks[pic,:,:,0])
    # #     np.add.at(indices[0], range(0,len(indices[0])), 1)
    # #     np.add.at(indices[1], range(0,len(indices[1])), 1)
    # #     print(np.shape(indices))
    # #     # indices = np.nonzero(mask_train)
    # #     # np.add.at(indices_train[0], range(0,len(indices_train[0])), 1)
    # #     # np.add.at(indices_train[1], range(0,len(indices_train[0])), 1)
    # #     por = images[pic,indices]
    # #     bakgrund = np.delete(images[pic],indices)
    # #
    # # raise
    # mu_por = test_img.mean()
    # median_por = np.median(test_img)
    # sigma_por = test_img.std()
    # textstr_por = '\n'.join((
    #     r'$\mu=%.2f$' % (mu_por, ),
    #     r'$\mathrm{median}=%.2f$' % (median_por, ),
    #     r'$\sigma=%.2f$' % (sigma_por, )))
    # #
    # # mu_bak = bakgrund_train.mean()
    # # median_bak = np.median(bakgrund)
    # # sigma_bak_ = bakgrund.std()
    # # textstr_bak_ = '\n'.join((
    # #     r'$\mu=%.2f$' % (mu_bak, ),
    # #     r'$\mathrm{median}=%.2f$' % (median_bak, ),
    # #     r'$\sigma=%.2f$' % (sigma_bak, )))
    # mu_bak = images.mean()
    # median_bak = np.median(images)
    # sigma_bak = images.std()
    # textstr_bak_ = '\n'.join((
    #     r'$\mu=%.2f$' % (mu_bak, ),
    #     r'$\mathrm{median}=%.2f$' % (median_bak, ),
    #     r'$\sigma=%.2f$' % (sigma_bak, )))
    # # raise
    # fig,axes=plt.subplots(1,2,figsize=(16,9))
    # sns.distplot(test_img.ravel(), kde=False, fit=stats.norm, ax=axes[0])
    # axes[0].set(xlabel='Intensity', ylabel='Density', title = 'Distribution of pores')
    # props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    # axes[0].text(0.05, 0.95, textstr_por, transform=axes[0].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)
    #
    # sns.distplot(images.ravel(), kde=False, fit=stats.norm, ax=axes[1], color="r")
    # axes[1].set(xlabel='Intensity', ylabel='Density', title = 'Distribution of background')
    # axes[1].text(0.05, 0.95, textstr_bak_, transform=axes[1].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)
    # fig.suptitle('Training set\n', fontsize=18, fontweight="bold")
    # fig.suptitle('Test set\n', fontsize=18, fontweight="bold")
    #
    #
    # # plt.hist(images.ravel(), bins = 255)
    # plt.tight_layout()
    #
    # plt.show()

if __name__ == '__main__':
    vis_data_dist()
