import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(1, '../processing')
from extract_data_ratio import extract_data_ratio

data_path = '../data/train_val_data_border_clean/'
test_path = '../data/test_data_border_clean/'

sns.set(color_codes=True)

def vis_data_dist_ratio(channels = 1):
    images22, _ = extract_data_ratio(data_path, channels, 22)
    test_img22, _ = extract_data_ratio(test_path, channels, 22)
    images22 =  np.hstack([images22.ravel(), test_img22.ravel()])

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

if __name__ == '__main__':
    vis_data_dist_ratio()
