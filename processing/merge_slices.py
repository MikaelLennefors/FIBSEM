import numpy as np
import os

from PIL import Image
from scipy.special import binom

load_path_test = '../data/test_data'
save_path_merge = '../data/train_val_data_merged/'
save_path_merge_test = '../data/test_data_merged/'

def merge_data(path_to_data = '../data/raw_data_no_mask/', channels = 7):

    og_dir = os.listdir(path_to_data)
    og_dir.sort()

    imgs = []

    for file_name in og_dir:
        img = Image.open(path_to_data + file_name)
        if 'image' in file_name:
            imgs.append(img.getdata())

    images = np.array(imgs)

    images = images.reshape((-1, 256, 256))

    test_dir = os.listdir(load_path_test)
    test_dir.sort()

    im_weights = []

    for i in range(channels):
        im_weights.append(binom(6, i))

    im_weights /= sum(im_weights)

    merged_im = 0
    n_images = np.shape(images)[0] // channels

    test_index = {22: [], 30: [], 45: []}

    for i in test_dir:
        if 'image' in i:
            test_index[int(i[5:7])].append(int(i[8:10]))

    for i in test_index.keys():
        for j in range(n_images // len(test_index)):
            merged_im = 0
            for k in range(channels):
                merged_im = merged_im + im_weights[k]*images[k+j*channels]

            merged_im = merged_im - np.quantile(merged_im, 0.1)
            merged_im = merged_im / np.max(merged_im)
            merged_im = np.clip(merged_im, 0, 1)
            merged_im *= 255

            im = Image.fromarray(merged_im.astype(np.uint8))
            if j not in test_index[i]:
                im.save(save_path_merge + 'image' + str(i) + '_' + str(j).zfill(2) + '.png')
            else:
                im.save(save_path_merge_test + 'image' + str(i) + '_' + str(j).zfill(2) + '.png')

if __name__ == '__main__':
     merge_data()