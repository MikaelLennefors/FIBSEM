from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from libtiff import TIFF, TIFFfile
path = '/home/sms/github/unet_data/'
# path = '../data/'

save_path = '/home/sms/github/unet_data/test_data/'
save_path = '../data/train_val_data_border_clean/'


def read_tiff(path):
    img1 = TIFF.open(path)
    images = []
    for image in img1.iter_images():
        tmp = np.around(image*(255/65535))
        images.append(tmp)

    return images

if __name__ == '__main__':
    ratio = input("Porositet: ")
    filename = 'regions_HPC{}.tif'.format(ratio)
    file = path + filename
    img = read_tiff(file)
    img64 = np.array([])
    print(np.shape(img))
    print(np.min(img[38]))

    patch_size = 64
    loop_end_patch = np.shape(img)[1] - patch_size

    for image in range(np.shape(img)[0]):
        curr_img = img[image]
        for rows in range(loop_end_patch):
            for col in range(loop_end_patch):
                tmp = curr_img[rows:rows+patch_size,col:col+patch_size]
                print(tmp)
                print(np.shape(tmp))
