import numpy as np
import cv2
import matplotlib.pyplot as plt



if __name__ == '__main__':
    path = '../data/magnus_data/train/image22_000.png'
    im = cv2.imread(path)
    print(np.shape(im))
    norma = im/255
    mu = np.mean(im)
    sig = np.std(im)
    stand = (im - mu)/sig
    stand_pos = (stand - np.min(stand))/(np.max(stand)-np.min(stand))
    print(stand_pos)
    print(np.min(stand_pos))
    print(np.max(stand_pos))
    median = cv2.medianBlur(stand_pos,5)
    plt.imshow(stand_pos, cmap = 'gray')
    # plt.imshow(norma, cmap = 'gray')
    # plt.imshow(median, cmap = 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.imshow(stand)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    # cv2.imshow('Normalized',norma)
    # cv2.imshow('Standardized',stand)
    # print(stand)
    # cv2.imwrite('norm_img.png', norma.astype(int))
    # cv2.imwrite('stand_img.png', stand.astype(int))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
