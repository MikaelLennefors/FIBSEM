# import numpy as np
import jax.numpy as np
from jax import jit
import time

@jit
def zca_whitening(img, epsilon = 1e-3):
    start_time = time.time()
    end_time = time.time()-start_time
    #print(end_time)
    x = np.shape(img)[1]
    y = np.shape(img)[2]
    channels = np.shape(img)[3]
    start_time = time.time()
    if channels > 1:
        img = np.swapaxes(img,2,3)
        img = np.swapaxes(img,1,2)
        img = img.reshape(-1,np.shape(img)[2],np.shape(img)[3],1)
    og_shape = np.shape(img)
    start_time = time.time()
    img = img.reshape(og_shape[0], -1)
    end_time = time.time()-start_time
    #print(end_time)

    img = img / 255.
    start_time = time.time()
    img = img - np.mean(img,axis=0)
    end_time = time.time()-start_time
    #print(end_time)
    start_time = time.time()
    U, S, V = np.linalg.svd(np.cov(img, rowvar=True))
    end_time = time.time()-start_time
    #print(end_time)
    start_time = time.time()
    img_ZCA = np.dot(np.dot(np.dot(U,np.diag(1.0/np.sqrt(S + epsilon))),U.T),img)
    end_time = time.time()-start_time
    #print(end_time)
    start_time = time.time()
    min_ZCA = np.min(img_ZCA)
    max_ZCA = np.max(img_ZCA)
    end_time = time.time()-start_time
    #print(end_time)

    img_ZCA = (img_ZCA - min_ZCA) / (max_ZCA - min_ZCA)
    start_time = time.time()
    img_ZCA = img_ZCA.reshape(og_shape)
    end_time = time.time()-start_time
    #print(end_time)
    start_time = time.time()
    if channels > 1:
        img_ZCA = img_ZCA.reshape(-1, channels, x, y, 1)
        img_ZCA = np.swapaxes(img_ZCA,1,2)
        img_ZCA = np.swapaxes(img_ZCA,2,3)
    end_time = time.time()-start_time
    #print(end_time)
    return img_ZCA
