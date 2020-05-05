import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(1, '../processing')
from extract_data import extract_data

data_path = '../data/train_val_data_border_clean/'
test_path = '../data/test_data_border_clean/'

sns.set(color_codes=True)

def vis_data_dist(channels = 1):
    images, masks = extract_data(data_path, channels)
    test_img, test_masks = extract_data(test_path, channels)

    indices = []
    indices_test = []
    por = np.array([])
    bakgrund = np.array([])
    por_test = np.array([])
    bakgrund_test = np.array([])

    for img in range(masks.shape[0]):
        loc_indices = np.nonzero(masks[img,:,:,0])
        np.add.at(loc_indices[0], range(0,len(loc_indices[0])), 1)
        np.add.at(loc_indices[1], range(0,len(loc_indices[1])), 1)
        indices.append(loc_indices)
        tmp = images[img,loc_indices[0], loc_indices[1], 0]
        por = np.append(por,tmp, axis=0)
        bak_temp = np.delete(images[img, :, :,0],indices[img])
        bakgrund = np.append(bakgrund, bak_temp)

    for img in range(test_masks.shape[0]):
        loc_indices = np.nonzero(test_masks[img,:,:,0])
        np.add.at(loc_indices[0], range(0,len(loc_indices[0])), 1)
        np.add.at(loc_indices[1], range(0,len(loc_indices[1])), 1)
        indices_test.append(loc_indices)
        tmp = test_img[img,loc_indices[0], loc_indices[1], 0]
        por_test = np.append(por_test,tmp, axis=0)
        bak_temp = np.delete(test_img[img, :, :,0],indices_test[img])
        bakgrund_test = np.append(bakgrund_test, bak_temp)

    means = np.zeros(4)
    stds = np.zeros(4)
    textstrs = [' ', ' ', ' ', ' ']
    name = ['Pores training and validation', 'Background training and validation', 'Pores testing', 'Background testing']
    col = ['b', 'r', 'g', 'y']
    data_sets = [por, bakgrund, por_test, bakgrund_test]

    fig,axes=plt.subplots(2,2,figsize=(24,13.5))
    axes = axes.flatten()
    fig.suptitle('Distribution of intensities', fontsize=18, fontweight="bold")
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

    for i in range(4):
        means[i] = data_sets[i].mean()
        stds[i] = data_sets[i].std()
        textstrs[i] = '\n'.join((
            r'$\mu=%.2f$' % (means[i], ),
            r'$\sigma=%.2f$' % (stds[i], )))
        sns.distplot(data_sets[i], kde=True, ax=axes[i], color=col[i])
        axes[i].set(xlabel='Intensity', ylabel='Density', title = name[i])
        axes[i].text(0.05, 0.95, textstrs[i], transform=axes[i].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()

if __name__ == '__main__':
    vis_data_dist()
