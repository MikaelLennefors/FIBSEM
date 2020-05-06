import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tabulate
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from GPyOpt.methods import BayesianOptimization

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam

sys.path.insert(1, './networks')
import losses
from unet import unet
from dunet import D_Unet
from multiresunet import MultiResUnet
from nestnet import Nest_Net

sys.path.insert(1, './processing')
from elastic_deformation import elastic_transform
from split_data import gen_data_split
from extract_data import extract_data
from whitening import zca_whitening
from split_grid import split_grid
from keras_augmentation import gen_aug

sys.path.insert(1, './visualization')
from keras_callbacks import PredictionCallback

channels = 1

gpu = 'Xp'

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        gpu = 'V'

print(gpu)

if len(sys.argv) > 2:
    channels = int(sys.argv[2])
if channels == 1:
    network = 'unet'
if channels > 1:
    network = 'dunet'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

if gpu == 'Xp':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
   gpu = 'Xp'

data_path = './data/train_val_data_border_clean/'
test_path = './data/test_data_border_clean/'

log_path = './results/{}/log.out'.format(gpu) #TODO Old from tensorboard
weights_path = './results/{}/weights'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

#TODO WILL WE HA PARAMTERS SÅ HÄR?
grid_split = 0
grid_split = 2**grid_split

NO_OF_EPOCHS = 20
max_count = 1
k_fold = 1

elast_deform = True
elast_alpha = 2
elast_sigma = elast_affine_alpha = 0.08

net_filters = 32
net_lr = 1e-5
net_bin_split = 0.3164
net_drop = 0.5
net_activ_fun = 0

prop_elastic = 0.05

bins = 2
resampling_const = 2


zca_coeff = 1e-2

max_intensity = 255.

aug_args = dict(
            vertical_flip = True,
            # horizontal_flip = True,
            # shear_range = 0.01,
            # rotation_range = 20,
            # zoom_range = 0.01,
            fill_mode = 'reflect'
        )

images, masks = extract_data(data_path, channels, standardize = False)
test_img, test_mask = extract_data(test_path, channels, standardize = False)

images_standardized, masks = extract_data(data_path, channels, standardize = True)
test_img_standardized, test_mask = extract_data(test_path, channels, standardize = True)

if grid_split > 1:
    images, masks = split_grid(images, masks, grid_split)
    test_img, test_mask = split_grid(test_img, test_mask, grid_split, test_set = True)

t_gen = []
t_gen_standardized = []
v_img = []
v_img_standardized = []
v_mask = []

for i in range(k_fold):
    train_images, train_mask, val_images, val_mask = gen_data_split(images, masks, random_seed = i)
    train_images_standardized, train_mask, val_images_standardized, val_mask = gen_data_split(images_standardized, masks, random_seed = i)
    if grid_split > 1:
        x = np.mean(train_mask, axis = (1,2,-1)) / max_intensity
        min_pics = np.shape(x)[0]
        img_poros = {}
        new_indices = []
        for j in range(bins):
            A = np.where(x >= (1/bins)*j)
            B = np.where(x < (1/bins)*(1+j))
            if j == (bins-1):
                B = np.where(x <= (1/bins)*(1+j))
            img_poros[j] = np.intersect1d(A, B)
            min_pics = np.min([len(img_poros[j]), min_pics])
        min_pics = resampling_const*min_pics

        for j in range(bins):
            curr_img = img_poros[j]
            if np.shape(curr_img)[0] != min_pics / resampling_const:
                indices = random.sample(list(curr_img), min_pics)
                new_indices.extend(indices)
            else:
                new_indices.extend(np.repeat(curr_img, resampling_const))
        new_indices = np.array(new_indices)
        train_images = train_images[new_indices]
        train_images_standardized = train_images_standardized[new_indices]
        train_mask = train_mask[new_indices]
    if channels > 1:
        train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], channels, 1)
        train_images_standardized = train_images_standardized.reshape(-1, np.shape(train_images_standardized)[1], np.shape(train_images_standardized)[2], channels, 1)
    else:
        train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], 1)
        train_images_standardized = train_images_standardized.reshape(-1, np.shape(train_images_standardized)[1], np.shape(train_images_standardized)[2], 1)
    train_mask = train_mask.reshape(-1, np.shape(train_mask)[1], np.shape(train_mask)[2], 1)

    aug_batch = np.shape(train_images)[0]

    train_gen = gen_aug(train_images, train_mask, aug_args, aug_batch)
    train_gen_standardized = gen_aug(train_images_standardized, train_mask, aug_args, aug_batch)

    t_gen.append(train_gen)
    t_gen_standardized.append(train_gen_standardized)
    v_img.append(val_images)
    v_img_standardized.append(val_images_standardized)
    v_mask.append(val_mask)

result_dict = []



iteration_count = 0

def evaluate_network(parameters):
    global iteration_count
    global test_img
    global result_dict
    parameters = parameters[0]
    net_drop = parameters[0]
    net_filters = int(parameters[1])
    net_lr = parameters[2]
    prop_elastic = parameters[3]
    b_size = int(parameters[4])
    preproc = int(parameters[5])

    zero_weight = np.mean(train_mask)
    mean_benchmark = []
    net_lr = math.pow(10,-net_lr)
    #print('drop:', net_drop, '\nfilters:', net_filters, '\nlr:', net_lr, '\nprop el:', prop_elastic, '\nb_size:', b_size, '\nwhitening:', whitening)
    sys.stdout.write("\rNumber of Bayesian iterations: {}".format(iteration_count))
    sys.stdout.flush()
    for i in range(k_fold):
        if preproc == 0:
            train_gen = t_gen_standardized[i]
            val_img = v_img_standardized[i]
            test_img = test_img_standardized
        else:
            train_gen = t_gen[i]
            val_img = v_img[i]
            test_img = test_img
        val_mask = v_mask[i]

        if channels == 1:
            input_size = np.shape(val_img)[1]
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        else:
            input_size = (np.shape(val_img)[1],channels)
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        m.compile(optimizer = Adam(lr = net_lr), loss = losses.bce_iou_loss(zero_weight), metrics = [losses.iou_coef, 'accuracy'])

        earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = NO_OF_EPOCHS // 2, mode = 'max')
        earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.6, patience = NO_OF_EPOCHS // 2)

        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(m.layers[0].get_weights()))
        callbacks_list = [earlystopping1, earlystopping2]
        # callbacks_list = [earlystopping1, earlystopping2, PredictionCallback(test_img, test_mask, callback_path)]

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

            if preproc == 1:
                x = x / max_intensity
                val_img = val_img / max_intensity
                test_img = test_img / max_intensity
            elif preproc > 1:
                whitening = 10**-(preproc - 1)
                x = zca_whitening(x, whitening)
                val_img = zca_whitening(val_img, whitening)
                test_img = zca_whitening(test_img, whitening)


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
                        y[j] = im_mask_t
            y = np.around(y)
            max_val = max(np.max(y), np.max(val_mask))
            if max_val > 1:
                raise
            results = m.fit(x, y, verbose = 0, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            # weights = []
            # from itertools import chain
            # for layer in m.layers:
            #     curr_weights = layer.get_weights()
            #     if curr_weights:
            #         #print(curr_weights)
            #         for countttt in range(2):
            #             weights.extend(np.ravel(list(chain.from_iterable(curr_weights))[countttt]))
            # print(np.min(weights))
            # print(np.max(weights))
            # print(np.mean(weights))
            # print(np.median(weights))
            # print(weights)
            #plt.hist(weights, bins = 100)
            # plt.show()
            # weights = weights.flatten()
            # print(np.shape(weights))

            count += 1
            if count >= max_count:
                break

        test = m.predict(test_img, verbose = 0)
        testy_mask = np.around(test).reshape(np.shape(test)[0],np.shape(test)[1],np.shape(test)[2]).astype(np.uint8)
        diff_mask = np.abs(testy_mask - np.squeeze(test_mask, axis = -1))
        pred = m.evaluate(test_img, test_mask, verbose = 0)
        summed_diff = np.sum(diff_mask, axis = 0)
        # plt.imshow(array_to_img(summed_diff.reshape(256,256,1)))
        # plt.show()
        # sys.exit()
        # im = Image.fromarray(testy_mask)
        # im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')
        score = pred[1]

        mean_benchmark.append(score)
    m1 = np.mean(mean_benchmark)


    pre_proc = {0: 'Standardized',
                1: 'Normalized',
                2: 'ZCA: 1e-1',
                3: 'ZCA: 1e-2',
                4: 'ZCA: 1e-3',
                5: 'ZCA: 1e-4',
                6: 'ZCA: 1e-5',
                7: 'ZCA: 1e-6'}

    result_dict.append({'Mean IoU': m1,
                    'Filters': net_filters,
                    'Learning rate': net_lr,
                    'Pre processing': pre_proc[preproc],
                    'Batch size': b_size,
                    'Dropout': net_drop,
                    'Elastic proportion': prop_elastic})
    iteration_count += 1

    #print('One result appended')
    #exit_print(result_dict)
    return m1

def main():
    bds = [{'name': 'net_drop', 'type': 'continuous', 'domain': (0.3, 0.5)},
            {'name': 'net_filters', 'type': 'discrete', 'domain': (16, 32, 64)},
            {'name': 'net_lr', 'type': 'continuous', 'domain': (3, 6)},
            {'name': 'prop_elastic', 'type': 'continuous', 'domain': (0, 0.2)},
            {'name': 'b_size', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 6)},
            {'name': 'pre_processing', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4, 5, 6, 7)}]

    optimizer = BayesianOptimization(f=evaluate_network,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.05,
                                     exact_feval=True,
                                     maximize=True)

    optimizer.run_optimization(max_iter=100)

    optimizer.plot_acquisition()
    optimizer.plot_convergence()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        exit_print(result_dict, gpu, network, channels)
