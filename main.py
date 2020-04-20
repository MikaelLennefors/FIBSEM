import numpy as np
import os
import random
import sys

# import statistics
import time
import math
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.optimizers import Adam

from bayes_opt import BayesianOptimization
from PIL import Image

sys.path.insert(1, './networks')
import losses
from unet import unet
from dunet import D_Unet
from multiresunet import MultiResUnet
from nestnet import Nest_Net, bce_dice_loss

sys.path.insert(1, './processing')
from elastic_deformation import elastic_transform
from split_data import gen_data_split
from datetime import datetime
from extract_data import extract_data
from whitening import zca_whitening
from split_grid import split_grid
from keras_augmentation import gen_aug

channels = 1

gpu = 'Xp'

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        gpu = 'V'
print(gpu)
if len(sys.argv) > 2:
    channels = int(sys.argv[2])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
if gpu == 'Xp':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# while int(os.environ["CUDA_VISIBLE_DEVICES"]) not in [0,1]:
#    os.environ["CUDA_VISIBLE_DEVICES"] = input("Choose GPU, 0 for Xp, 1 for V: ")
#
# channels = int(input("Choose channels, 1, 3, 5 or 7: "))
# while channels not in [1,3,5,7]:
#     channels = int(input("Choose channels, 1, 3, 5 or 7: "))


if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
   gpu = 'Xp'

data_path = './data/train_val_data_border_clean/'
test_path = './data/test_data_border_clean/'
log_path = './results/{}/log.out'.format(gpu)
weights_path = './results/{}/weights'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

grid_split = 2
grid_split = 2**grid_split


NO_OF_EPOCHS = 200
aug_batch = grid_split*grid_split*180
max_count = 3
b_size = 128
elast_deform = True
elast_alpha = 2
elast_sigma = 0.08
elast_affine_alpha = 0.08
net_filters = 32
prop_elastic = 0.05
net_lr = 1e-5
net_bin_split = 0.3164
net_drop = 0.5
net_activ_fun = 0



aug_args = dict(
            vertical_flip = True,
            horizontal_flip = True,
            #shear_range = 0.01,
            #rotation_range = 20,
            #zoom_range = 0.01,
            fill_mode = 'reflect'
        )

zca_coeff = 5e-2

# with open('results/{}/results.txt'.format(gpu), 'w') as f:
#         for key in configurations[0].keys():
#             f.write('%s\t' % key)
#         f.write('val_iou\t')
#         f.write('val_accuracy\t')
#         f.write('val_TP\t')
#         f.write('val_TN\t')
#         f.write('val_FP\t')
#         f.write('val_FN')
images, masks = extract_data(data_path, channels)
test_img, test_mask = extract_data(test_path, channels)

images, masks = split_grid(images, masks, grid_split)
test_img, test_mask = split_grid(test_img, test_mask, grid_split)

#test_img = zca_whitening(test_img, zca_coeff)

test_img = np.array(test_img)
test_mask = test_mask / 255.

t_gen = []
v_img = []
v_mask = []
bins = 2


for i in range(3):
    print("nu går vi in i gen_data_split nr", i)
    train_images, train_mask, b, c = gen_data_split(images, masks, whitening_coeff = zca_coeff)

    test = np.linspace(0, 100, 101, dtype=np.uint8)

    x = np.mean(train_mask, axis = (1,2,-1))/255
    min_pics = np.shape(x)[0]
    img_poros = {}
    new_imgs = []
    new_masks = []
    now_time = time.time()
    for j in range(bins):
        A = np.where(x >= (1/bins)*j)
        B = np.where(x < (1/bins)*(1+j))
        if j == (bins-1):
            B = np.where(x <= (1/bins)*(1+j))
        img_poros[j] = np.intersect1d(A, B)
        min_pics = np.min([len(img_poros[j]), min_pics])

    for j in range(bins):
        curr_img = img_poros[j]
        if np.shape(curr_img)[0] != min_pics:
            indices = random.sample(list(curr_img), min_pics)
            new_imgs.append(indices)
            new_masks.append(indices)
        else:
            new_imgs.append(curr_img)
            new_masks.append(curr_img)

    new_imgs = np.array(new_imgs)
    new_imgs = new_imgs.flatten()
    new_masks = np.array(new_masks)
    new_masks = new_masks.flatten()
    train_images = train_images[new_imgs]
    train_mask = train_mask[new_masks]
    print(np.mean(train_mask)/255)
    train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], 1)
    train_mask = train_mask.reshape(-1, np.shape(train_mask)[1], np.shape(train_mask)[2], 1)
    print(np.shape(train_images))
    print(np.shape(train_mask))
    train_images = zca_whitening(train_images, zca_coeff)
    a = gen_aug(train_images, train_mask, aug_args, aug_batch)
    t_gen.append(a)
    b = zca_whitening(b)
    v_img.append(np.array(b))
    v_mask.append(c)

# if len(sys.argv) <= 3:
#     print("Disabling debug")
#     import logging
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
#     logging.getLogger('tensorflow').setLevel(logging.FATAL)


def evaluate_network(net_drop, net_filters, net_lr, prop_elastic):
    mean_benchmark = []
    net_lr = math.pow(10,-net_lr)
    net_filters = int(math.pow(2,math.floor(net_filters)+4))
    print('drop: ', net_drop, '\nfilters: ', net_filters, '\nlr: ', net_lr, '\nprop el: ', prop_elastic)
    # print(net_lr)
    # print(net_filters)
    # raise
    for i in range(3):
        train_gen = t_gen[i]
        val_img = v_img[i]
        val_mask = v_mask[i]

        if channels == 1:
            input_size = np.shape(val_img)[1]
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        else:
            input_size = (np.shape(val_img)[1],channels)
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy'])



        earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = 100, mode = 'max')
        earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.6, patience = 50)
        class PredictionCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
                # new_shape = int(np.shape(predic_mask)/gridsplit)
                testy_mask = np.around(predic_mask).reshape(np.shape(predic_mask)[1],np.shape(predic_mask)[1]).astype(np.uint8)*255

                im = Image.fromarray(testy_mask)
                im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')


        callbacks_list = [earlystopping1, earlystopping2]

        count = 0
        for c in train_gen:
            y = np.array(c[-1])
            x = []
            for j in range(len(c) - 1):
                x.append(c[j])
            x = np.array(x)

            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)
            if np.shape(x)[3] == 1:
                x = np.squeeze(x, axis = 3)
            if elast_deform == True:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge_t = elastic_transform(img, img.shape[1] * elast_alpha, img.shape[1] * elast_sigma, img.shape[1] * elast_affine_alpha, random_state = seed)
                            if channels > 1:
                                x[j,:,:,k] = im_merge_t
                            else:
                                x[j,:,:,k] = im_merge_t.reshape(img.shape[0],img.shape[1])
                        mask = y[j].copy().reshape(np.shape(train_mask)[1],np.shape(train_mask)[1])
                        seed = np.random.RandomState(randoint)
                        im_mask_t = elastic_transform(mask, mask.shape[1] * elast_alpha, mask.shape[1] * elast_sigma, mask.shape[1] * elast_affine_alpha, random_state = seed)
                        #print(np.shape(im_mask_t))
                        # im_mask_t = im_merge_t[...,0]
                        y[j] = im_mask_t#.reshape(256,256,1)
            y = np.around(y / 255.)

            results = m.fit(x, y, verbose = 1, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            # i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            # i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            # i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            # i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            # i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            # i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

            count += 1
            #print(max_count - count)
            if count >= max_count:
                break
        pred = m.evaluate(test_img, test_mask, verbose = 0)
        score = pred[1]

        mean_benchmark.append(score)
    m1 = np.mean(mean_benchmark)
    # mdev = np.std(mean_benchmark)
    return m1
# test = evaluate_network(4,2)



pbounds = {'net_drop': (0.0,0.5),
    'net_filters': (0.0, 3.0),
    'net_lr': (1.0, 3.0),
    'prop_elastic': (0.0, 0.3)
    }

optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,
)

start_time = time.time()
optimizer.maximize(init_points=10, n_iter=100,)
time_took = time.time() - start_time

# print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)
