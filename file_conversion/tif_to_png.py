from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from libtiff import TIFF, TIFFfile
import scipy.misc
# path = '/home/sms/github/unet_data/'
path = '../data/'

# save_path = '/home/sms/github/unet_data/test_data/'
save_path = '../data/all_data_384/'


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
    print(np.shape(img))
    save_name = 'image{}_'.format(ratio)

    for image in range(np.shape(img)[0]):
        curr_img = img[image]
        curr_img.astype(int)
        cv2.imwrite(save_path + save_name + str(image).zfill(3) + '.png', curr_img)
