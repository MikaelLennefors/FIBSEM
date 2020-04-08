import numpy as np
import time
#import scipy.linalg
#import jax.numpy as np
#import jax.scipy as jsp
#from numpy.linalg import multi_dot
def zca_whitening(img, epsilon = 1e-3):
    print("--"*20)
    start_time = time.time()
    channels = np.shape(img)[3]
    if channels > 1:
        img = np.swapaxes(img,2,3)
        img = np.swapaxes(img,1,2)
        img = img.reshape(-1,np.shape(img)[2],np.shape(img)[3],1)
    og_shape = np.shape(img)
    stop_time = time.time() - start_time
    print(stop_time)
    img = img.reshape(og_shape[0], -1)

    img = img / 255.

    img = img - img.mean(axis=0)
    #start_time = time.time()
    co = np.cov(img, rowvar=True)
    #cov_time = time.time() - start_time
    #start_time2 = time.time()
    U, S, V = np.linalg.svd(co)
    #svd_time = time.time() - start_time2
    #print("svd time:",svd_time)
    #print("svd over cov:", svd_time/cov_time)
    stop_time = time.time() - start_time
    print(stop_time)

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
    stop_time = time.time() - start_time
    print(stop_time)
    if channels > 1:
        img_ZCA = np.array(np.split(img_ZCA, np.shape(img_ZCA)[0]/channels))
        img_ZCA = np.swapaxes(img_ZCA,1,2)
        img_ZCA = np.swapaxes(img_ZCA,2,3)
    stop_time = time.time() - start_time
    print(stop_time)

    return img_ZCA
