import numpy as np
import os

from PIL import Image
from scipy.special import binom

load_path_test = '../data/test_data'
save_path_merge = '../data/train_val_data_merged/'
save_path_merge_test = '../data/test_data_merged/'

def merge_data(path_to_data = '../data/raw_data/', channels = 7):

    og_dir = os.listdir(path_to_data)
    og_dir.sort()

    test_dir = os.listdir(load_path_test)
    test_dir.sort()

    test_index = {22: [], 30: [], 45: []}

    for i in test_dir:
        if 'image' in i:
            test_index[int(i[5:7])].append(int(i[8:10]))

    t_i = ''
    for i in test_index.keys():
        for j in range(100):
            if j in test_index[i]:
                for k in np.linspace(channels*j,channels*j+channels - 1, channels, dtype=np.uint16):
                    t_i = 'image' + str(i) + '_' + str(k).zfill(3) + '.png'
                    os.rename(path_to_data + t_i, '../data/raw_test_data/' + t_i)

if __name__ == '__main__':
     merge_data()