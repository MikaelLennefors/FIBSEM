# import numpy as np
import time
#import scipy.linalg
import jax.numpy as np
# from jax import device_put
# import jax.scipy as np
#from numpy.linalg import multi_dot
def zca_whitening(img, epsilon = 1e-3):
    print("--"*20)
    start_time = time.time()
    x = np.shape(img)[1]
    y = np.shape(img)[2]
    channels = np.shape(img)[3]
    if channels > 1:
        img = np.swapaxes(img,2,3)
        img = np.swapaxes(img,1,2)
        img = img.reshape(-1,np.shape(img)[2],np.shape(img)[3],1)
    og_shape = np.shape(img)
    # stop_time = time.time() - start_time
    # print('1 reshape\t', stop_time)
    start_time = time.time()
    img = img.reshape(og_shape[0], -1)

    img = img / 255.

    img = img - img.mean(axis=0)
    # img = device_put(img)
    #start_time = time.time()
    # co = device_put(co)
    #cov_time = time.time() - start_time
    #start_time2 = time.time()
    U, S, V = np.linalg.svd(np.cov(img, rowvar=True))
    #svd_time = time.time() - start_time2
    #print("svd time:",svd_time)
    #print("svd over cov:", svd_time/cov_time)
    # stop_time = time.time() - start_time
    # print('cov+svd\t', stop_time)
    # start_time = time.time()

    #start_time = time.time()
    #img_ZCA = multi_dot([U,np.diag(1.0/np.sqrt(S + epsilon)),U.T,img])
    #img_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(img)
    # print(np.shape(U))
    # print(np.shape(np.diag(1.0/np.sqrt(S + epsilon))))
    # print(np.shape(U.T))
    # print(np.shape(img))
    # raise
    img_ZCA = np.dot(np.dot(np.dot(U,np.diag(1.0/np.sqrt(S + epsilon))),U.T),img)
    #dot_time = time.time() - start_time
    #print("dot time:",dot_time)

    min_ZCA = img_ZCA.min()
    max_ZCA = img_ZCA.max()

    img_ZCA = (img_ZCA - min_ZCA) / (max_ZCA - min_ZCA)

    img_ZCA = img_ZCA.reshape(og_shape)
    # stop_time = time.time() - start_time
    # print('dot\t', stop_time)
    # start_time = time.time()
    if channels > 1:
        # img_ZCA = np.split(img_ZCA, np.shape(img_ZCA)[0]/channels)
        img_ZCA = img_ZCA.reshape(-1, channels, x, y, 1)
        # stop_time = time.time() - start_time
        # print('split\t', stop_time)
        # start_time = time.time()
        # img_ZCA = np.array(img_ZCA)
        # stop_time = time.time() - start_time
        # print('to array\t', stop_time)
        img_ZCA = np.swapaxes(img_ZCA,1,2)
        # start_time = time.time()
        img_ZCA = np.swapaxes(img_ZCA,2,3)
        # start_time = time.time()
    # stop_time = time.time() - start_time
    # print('2 reshape\t', stop_time)


    return img_ZCA
