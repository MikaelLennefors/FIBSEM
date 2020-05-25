import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tabulate
import pandas as pd
import time
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from itertools import chain
from GPyOpt.methods import BayesianOptimization

from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam

sys.path.insert(1, './networks')
import losses
from unet import unet
from dunet import D_Unet
from multiresunet import MultiResUnet
from nestnet import Nest_Net
from early_stop import EarlyStoppingBaseline, EarlyStoppingDelta

sys.path.insert(1, './processing')
from elastic_deformation import elastic_transform
from split_data import gen_data_split
from extract_data import extract_data
from whitening import zca_whitening
from split_grid import split_grid
from keras_augmentation import gen_aug

sys.path.insert(1, './visualization')
from keras_callbacks import PredictionCallback
from exit_print import exit_print

channels = 1

gpu = 'Xp'

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        gpu = 'V'

if len(sys.argv) > 2:
    channels = int(sys.argv[2])
net = 0
if len(sys.argv) > 3:
    net = int(sys.argv[3])

if net == 0:
    network = 'unet'
if net == 1:
    network = 'dunet'
if net == 2:
    network = 'multiresunet'
if net == 3:
    network = 'nestnet'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

if gpu == 'Xp':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
   gpu = 'Xp'

train_path = './data/magnus_data/train/'
val_path = './data/magnus_data/val/'
test_path = './data/magnus_data/test/'

log_path = './results/{}/log.out/'.format(gpu) #TODO Old from tensorboard
weights_path = './results/{}/weights/'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

max_hours = 48

#TODO WILL WE HA PARAMTERS SÅ HÄR?
grid_split = 0
grid_split = 2**grid_split

NO_OF_EPOCHS = 48
max_count = 16
k_fold = 3

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
            horizontal_flip = True,
            rotation_range = 20,
            fill_mode = 'reflect'
        )

end_time = time.time() + max_hours*60*60
current_time = time.strftime("%H:%M:%S", time.localtime())
print('Current time:', current_time, '\nGPU:', gpu, '\nNetwork:', network, '\nChannels:', channels,'\nNumber of epochs:', NO_OF_EPOCHS, '\nMax count:', max_count, '\nk fold:', k_fold, '\nGrid split:', grid_split)

train_images, train_masks = extract_data(train_path, channels, standardize = False)
val_images, val_masks = extract_data(val_path, channels, standardize = False)
test_images, test_masks = extract_data(test_path, channels, standardize = False)

train_images_standardized, train_masks_standardized = extract_data(train_path, channels, standardize = True)
val_images_standardized, val_masks_standardized = extract_data(val_path, channels, standardize = True)
test_images_standardized, test_masks_standardized = extract_data(test_path, channels, standardize = True)

if grid_split > 1:
    images, masks = split_grid(images, masks, grid_split)
    test_images, test_masks = split_grid(test_images, test_masks, grid_split, test_set = True)
    images_standardized, _ = split_grid(images_standardized, masks_standardized, grid_split)
    test_images_standardized, _ = split_grid(test_images_standardized, test_masks_standardized, grid_split, test_set = True)

t_gen = []
t_gen_standardized = []
v_img = []
v_img_standardized = []
v_mask = []

for i in range(k_fold):
    # train_images, train_masks, val_images, val_masks = gen_data_split(images, masks, random_seed = i)
    # train_images_standardized, train_masks, val_images_standardized, val_masks = gen_data_split(images_standardized, masks, random_seed = i)
    if grid_split > 1:
        x = np.mean(train_masks, axis = (1,2,-1))
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
        train_masks = train_masks[new_indices]
    train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], channels, 1)
    train_images_standardized = train_images_standardized.reshape(-1, np.shape(train_images_standardized)[1], np.shape(train_images_standardized)[2], channels, 1)

    train_masks = train_masks.reshape(-1, np.shape(train_masks)[1], np.shape(train_masks)[2], 1)

    aug_batch = np.shape(train_images)[0]

    train_generator = gen_aug(train_images, train_masks, aug_args, aug_batch)
    train_generator_standardized = gen_aug(train_images_standardized, train_masks, aug_args, aug_batch)

    t_gen.append(train_generator)
    t_gen_standardized.append(train_generator_standardized)
    v_img.append(val_images)
    v_img_standardized.append(val_images_standardized)
    v_mask.append(val_masks)

result_dict = []
metric_dict = {'iou_coef': {},
               'val_iou_coef': {},
               'predicted_prop': {}}

v_img = np.array(v_img)
iteration_count = 0

def evaluate_network(parameters):
    k_fold_count = 0
    global iteration_count
    global test_images
    global result_dict

    net_drop = parameters['net_drop']
    net_filters = parameters['net_filters']
    net_lr = parameters['net_lr']
    prop_elastic = parameters['prop_elastic']
    b_size = int(parameters['b_size'])
    preproc = parameters['pre_processing']

    print('net_drop:', net_drop)
    print('net_filters:', net_filters)
    print('net_lr:', net_lr)
    print('prop_elastic:', prop_elastic)
    print('b_size:', b_size)
    print('preproc:', preproc)

    zero_weight = np.mean(train_masks)
    mean_benchmark = []
    train_gen = t_gen
    if preproc['type'] == 'Standardize':
        train_gen = t_gen_standardized
        val_img = v_img_standardized
        test_img = test_images_standardized
    elif preproc['type'] == 'Normalize':
        val_img = v_img / max_intensity
        test_img = test_images / max_intensity
    elif preproc['type'] == 'ZCA':
        val_img = []
        for i in range(k_fold):
            val_img.append(zca_whitening(v_img[i], preproc['whitening']))
            test_img = zca_whitening(test_images, preproc['whitening'])



    for data in zip(train_gen, val_img, v_mask):
        k_fold_count += 1
        metric_dict['iou_coef'][str(k_fold_count)] = []
        metric_dict['val_iou_coef'][str(k_fold_count)] = []
        metric_dict['predicted_prop'][str(k_fold_count)] = []
        train = data[0]
        validation_img = data[1]
        validation_mask = data[2]
        input_size = np.shape(train_images)[1:]
        if network == 'unet':
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, dout = net_drop)
        elif network == 'dunet':
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, dout = net_drop)
        elif network == 'multiresunet':
            m = MultiResUnet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, dout = net_drop)
        elif network == 'nestnet':
            m = Nest_Net(input_size = input_size, multiple = net_filters, activation = net_activ_fun, dout = net_drop)


        m.compile(optimizer = Adam(lr = net_lr), loss = losses.bce_iou_loss(zero_weight), metrics = [losses.iou_coef, 'accuracy'])

        # earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = NO_OF_EPOCHS // 2, mode = 'max')
        # earlystopping2 = EarlyStopping(monitor = 'val_accuracy', baseline = 0.6, patience = NO_OF_EPOCHS // 2,  mode = 'auto')
        patience = NO_OF_EPOCHS // 2
        callbacks_list = [PredictionCallback(test_img)]
        # callbacks_list = [earlystopping1, earlystopping2, PredictionCallback(test_img, test_mask, callback_path)]

        count = 0

        for c in train:
            y = np.array(c[-1])
            x = []
            for j in range(len(c) - 1):
                x.append(c[j])
            x = np.array(x)

            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)
            if preproc['type'] == 'Normalize':
                x = x / max_intensity
            elif preproc['type'] == 'ZCA':
                x = zca_whitening(x, preproc['whitening'])
            if prop_elastic > 0:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge_t = elastic_transform(img, img.shape[1] * elast_alpha, img.shape[1] * elast_sigma, img.shape[1] * elast_affine_alpha, random_state = seed)
                            x[j,:,:,k] = im_merge_t
                        mask = y[j].copy().reshape(np.shape(train_masks)[1],np.shape(train_masks)[1])
                        seed = np.random.RandomState(randoint)
                        im_mask_t = elastic_transform(mask, mask.shape[1] * elast_alpha, mask.shape[1] * elast_sigma, mask.shape[1] * elast_affine_alpha, random_state = seed)
                        y[j] = im_mask_t
            y = np.around(y)
            max_val = max(np.max(y), np.max(validation_mask))
            if max_val > 1:
                raise

            # if time.time() > end_time:
            #     raise KeyboardInterrupt
            # print("hi")
            results = m.fit(x, y, verbose = 2, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(validation_img, validation_mask), callbacks = callbacks_list)

            metric_dict['iou_coef'][str(k_fold_count)].extend(results.history['iou_coef'])
            metric_dict['val_iou_coef'][str(k_fold_count)].extend(results.history['val_iou_coef'])
            print(metric_dict)
            count += 1
            if count >= max_count:
                break

        # test = m.predict(test_img_curr, verbose = 0)
        # testy_mask = np.around(test).reshape(np.shape(test)[0],np.shape(test)[1],np.shape(test)[2]).astype(np.uint8)
        # diff_mask = np.abs(testy_mask - np.squeeze(test_mask, axis = -1))


        # weights = []
        # for layer in m.layers:
        #     curr_weights = layer.get_weights()
        #     if curr_weights:
        #         for countttt in list(chain.from_iterable(curr_weights)):
        #             weights.extend(np.ravel(countttt))
        # np.savetxt(weights_path + network + '_' + str(channels) + '_weights.txt2', weights)
        metric_dict['predicted_prop'][str(k_fold_count)].extend(callbacks_list[-1].predicted_prop)
        pred = m.evaluate(test_img, test_masks, batch_size = 6, verbose = 2)
        # print(pred)
        score = pred[1]

        mean_benchmark.append(score)
    df_test1 = pd.DataFrame(metric_dict['iou_coef'])
    df_test2 = pd.DataFrame(metric_dict['val_iou_coef'])
    df_test3 = pd.DataFrame(metric_dict['predicted_prop'])
    df_test1.to_csv(callback_path + network + '_' + str(channels) + '_iou_hist.txt', index = False)
    df_test2.to_csv(callback_path + network + '_' + str(channels) + '_val_iou_hist.txt', index = False)
    df_test3.to_csv(callback_path + network + '_' + str(channels) + '_test_proportions.txt', index = False)
    # predic_mask = m.predict(np.expand_dims(test_img[30], axis = 0))
    # testy_mask = 255. * np.around(predic_mask).reshape(np.shape(predic_mask)[1],np.shape(predic_mask)[1]).astype(np.uint8)
    # print("True proportion:", np.mean(test_masks[30]), "Predicted proportion:", np.mean(testy_mask) / 255.)

    # cv2.imwrite(callback_path + network + '_' + str(channels) + '_pred_mask' + '.png', testy_mask)
    m1 = np.mean(mean_benchmark)
    # np.savetxt(callback_path + network + '_' + str(channels) + '_iou_hist.txt', metric_dict['iou_coef'])
    # np.savetxt(callback_path + network + '_' + str(channels) + '_val_iou_hist.txt', metric_dict['val_iou_coef'])
    # print(metric_dict)
    iteration_count += 1

    result_dict.append({'Iteration': iteration_count,
                    'Mean IoU': m1,
                    'Filters': net_filters,
                    'Learning rate': net_lr,
                    'Pre processing': preproc['type'],
                    'Whitening': preproc['whitening'],
                    'Batch size': b_size,
                    'Dropout': net_drop,
                    'Elastic proportion': prop_elastic})


    #print('One result appended')
    #exit_print(result_dict)
    return mean_benchmark

N_FOLDS = 10
MAX_EVALS = 500
def main():
    parameters = {'unet': {'net_drop': 0.4,
                  'net_filters': 64,
                  'net_lr': 6.4e-5,
                  'prop_elastic': 0.04,
                  'b_size': 6,
                  'pre_processing': {'type': 'Normalize',
                                     'whitening': 0}},
                  'dunet3': {'net_drop': 0.45,
                                'net_filters': 64,
                                'net_lr': 5e-5,
                                'prop_elastic': 0.05,
                                'b_size': 5,
                                'pre_processing': {'type': 'Standardize',
                                                   'whitening': 0}},
                  'dunet5': {'net_drop': 0.45,
                                'net_filters': 32,
                                'net_lr': 5e-5,
                                'prop_elastic': 0.05,
                                'b_size': 4,
                                'pre_processing': {'type': 'Standardize',
                                                   'whitening': 0}},
                  'multiresunet': {'net_drop': 0.5,
                                'net_filters': 64,
                                'net_lr': 2.5e-5,
                                'prop_elastic': 0.17,
                                'b_size': 3,
                                'pre_processing': {'type': 'Normalize',
                                                   'whitening': 0}},
                  'multiresunet3': {'net_drop': 0.35,
                                'net_filters': 64,
                                'net_lr': 2.5e-4,
                                'prop_elastic': 0.125,
                                'b_size': 4,
                                'pre_processing': {'type': 'ZCA',
                                                   'whitening': 1e-1}},
                  'multiresunet5': {'net_drop': 0.35,
                                'net_filters': 64,
                                'net_lr': 1e-4,
                                'prop_elastic': 0.125,
                                'b_size': 1,
                                'pre_processing': {'type': 'Normalize',
                                                   'whitening': 0}},
                  'nestnet': {'net_drop': 0.365,
                                'net_filters': 64,
                                'net_lr': 1e-4,
                                'prop_elastic': 0.11,
                                'b_size': 2,
                                'pre_processing': {'type': 'Standardize',
                                                   'whitening': 0}},
                  'nestnet3': {'net_drop': 0.4,
                                'net_filters': 64,
                                'net_lr': 5e-5,
                                'prop_elastic': 0.095,
                                'b_size': 2,
                                'pre_processing': {'type': 'Standardize',
                                                   'whitening': 0}},
                  'nestnet5': {'net_drop': 0.4,
                                'net_filters': 64,
                                'net_lr': 5e-5,
                                'prop_elastic': 0.095,
                                'b_size': 2,
                                'pre_processing': {'type': 'Standardize',
                                                   'whitening': 0}}}


    if channels > 1:
        curr_pars = parameters[network + str(channels)]
    else:
        curr_pars = parameters[network]


    print(evaluate_network(curr_pars))
    exit_print(result_dict, gpu, network, channels)
    # space = {
    #     'net_drop': hp.uniform('net_drop', 0.3, 0.5),
    #     'net_filters': hp.choice('net_filters', [32, 64]),
    #     'net_lr': hp.loguniform('net_lr', -math.log(10)*4.6, -math.log(10)*3.2),
    #     'prop_elastic': hp.uniform('prop_elastic', 0, 0.2),
    #     'b_size': hp.quniform('b_size', 1, 6, 1),
    #     'pre_processing': hp.choice('pre_processing', [{'type': 'Standardize', 'whitening': 0},
    #                                                     {'type': 'Normalize', 'whitening': 0},
    #                                                     {'type': 'ZCA',
    #                                                     'whitening': hp.loguniform('whitening', -math.log(10)*4, -math.log(10))}])
    # }
    # tpe_algorithm = tpe.suggest
    #
    # bayes_trials = Trials()
    #
    # best = fmin(fn = evaluate_network, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        exit_print(result_dict, gpu, network, channels)
    # except:
    #     print('Error')
    #     exit_print(result_dict, gpu, network, channels)
