import numpy as np
import os
import param
import random
import sys
import logging
import statistics
import time
import math
if len(sys.argv) <= 3:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

#from tensorflow.keras.utils import multi_gpu_model
from bayes_opt import BayesianOptimization
from PIL import Image
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.optimizers import Adam, SGD
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

gpus = tf.config.list_physical_devices('GPU')

if len(gpus) == 1:
    gpu = 'Xp'
else:
    gpu = 'V'
channels = 1

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        gpu = 'V'
if len(sys.argv) > 2:
    channels = int(sys.argv[2])

print(gpu)
print(channels)
raise
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# while int(os.environ["CUDA_VISIBLE_DEVICES"]) not in [0,1]:
#    os.environ["CUDA_VISIBLE_DEVICES"] = input("Choose GPU, 0 for Xp, 1 for V: ")

# channels = int(input("Choose channels, 1, 3, 5 or 7: "))
# while channels not in [1,3,5,7]:
#     channels = int(input("Choose channels, 1, 3, 5 or 7: "))


# if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
#    gpu = 'Xp'

data_path = './data/train_val_data_border/'
test_path = './data/test_data_border/'
log_path = './results/{}/log.out'.format(gpu)
weights_path = './results/{}/weights'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

params = param.HyperParam()

configurations = [params.generate_hyperparameters() for _ in range(10000)]

grid_split = 0


NO_OF_EPOCHS = 200
aug_batch = 180
max_count = 3
b_size = 1
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
with open('results/{}/results.txt'.format(gpu), 'w') as f:
        for key in configurations[0].keys():
            f.write('%s\t' % key)
        f.write('val_iou\t')
        f.write('val_accuracy\t')
        f.write('val_TP\t')
        f.write('val_TN\t')
        f.write('val_FP\t')
        f.write('val_FN')

images, masks = extract_data(data_path, channels)
test_img, test_mask = extract_data(test_path, channels)

test_img = zca_whitening(test_img, zca_coeff)
test_mask = test_mask / 255.

t_gen = []
v_img = []
v_mask = []
mean_benchmark = []
for i in range(3):
    a, b, c = gen_data_split(images, masks, channels = channels, b_size = aug_batch, whitening_coeff = zca_coeff, maskgen_args = aug_args)
    t_gen.append(a)
    v_img.append(b)
    v_mask.append(c)
def evaluate_network(net_drop, net_filters, net_lr, prop_elastic):
    net_lr = math.pow(10,-net_lr)
    net_filters = int(math.pow(2,math.floor(net_filters)+4))
    # print(net_lr)
    # print(net_filters)
    # raise
    for i in range(3):
        train_gen = t_gen[i]
        val_img = v_img[i]
        val_mask = v_mask[i]

        if channels == 1:
            input_size = np.shape(val_img)[1]//(2**grid_split)
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        else:
            input_size = (np.shape(val_img)[1]//(2**grid_split),channels)
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy'])



        earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = 100, mode = 'max')
        earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.6, patience = 50)
        # class PredictionCallback(tf.keras.callbacks.Callback):
        #     def on_epoch_end(self, epoch, logs={}):
        #         predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
        #         testy_mask = np.around(predic_mask).reshape(256,256).astype(np.uint8)*255

        #         im = Image.fromarray(testy_mask)
        #         im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')


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
                        mask = y[j].copy().reshape(256,256)
                        seed = np.random.RandomState(randoint)
                        im_mask_t = elastic_transform(mask, mask.shape[1] * elast_alpha, mask.shape[1] * elast_sigma, mask.shape[1] * elast_affine_alpha, random_state = seed)
                        #print(np.shape(im_mask_t))
                        # im_mask_t = im_merge_t[...,0]
                        y[j] = im_mask_t#.reshape(256,256,1)
            y = np.around(y / 255.)

            results = m.fit(x, y, verbose = 0, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
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
        mdev = np.std(mean_benchmark)
    return m1
# test = evaluate_network(4,2)



pbounds = {'net_drop': (0.0,0.5),
    'net_filters': (0.0, 3.0),
    'net_lr': (2.0, 5.0),
    'prop_elastic': (0.0, 0.3)
    }

optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

start_time = time.time()
optimizer.maximize(init_points=10, n_iter=100,)
time_took = time.time() - start_time

# print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)
# raise
# for i in configurations:


#     # net_filters = int(i['net_filters'])
#     # prop_elastic = i['prop_elastic']
#     # # net_activ_fun = int(i['net_activ_fun'])
#     # net_lr = i['net_lr']
#     # # net_bin_split = i['net_bin_split']
#     # net_drop = i['net_drop']
#     #zca_coeff = i['zca_coeff']
#     # i['val_iou'] = 0
#     # i['val_accuracy'] = 0
#     # i['val_TP'] = 0
#     # i['val_TN'] = 0
#     # i['val_FP'] = 0
#     # i['val_FN'] = 0

#     # conf_name = 'logs/{}_{}_{:.2e}_{:.2f}_{}_{:.2f}_{:.2f}_{}'.format(channels, net_filters, net_lr, net_bin_split, net_activ_fun,
#     #     net_drop, prop_elastic, datetime.now().strftime('%m-%d_%H:%M:%S'))
#     #print(np.shape(val_img))
#     #raise
#     if channels == 1:
#         input_size = np.shape(val_img)[1]//(2**grid_split)
#         m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
#         #m = Nest_Net(input_size, color_type=1, num_class=1, deep_supervision=False)
#         #m = MultiResUnet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, bin_weight = net_bin_split, dout = net_drop)
#     else:
#         input_size = (np.shape(val_img)[1]//(2**grid_split),channels)
#         m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
#     m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy', losses.TP, losses.TN, losses.FP, losses.FN])
#     #checkpoint = ModelCheckpoint(weights_path, monitor='val_iou_coef',
#     #                             verbose=1, save_best_only=True, mode='max')

#     #csv_logger = CSVLogger(log_path, append=True, separator=';')

#     #tensorboard = TensorBoard(log_dir = conf_name)

#     earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.001, patience = 100, mode = 'max')
#     earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.5, patience = 30, mode = 'max')
#     class PredictionCallback(tf.keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs={}):
#             predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
#             testy_mask = np.around(predic_mask).reshape(256,256).astype(np.uint8)*255

#             #diff_mask = np.abs(testy_mask - y[i].reshape(256,256)).astype(np.uint8)*255
#             im = Image.fromarray(testy_mask)
#             im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')
#             #im2.save(pred_path + 'diff_mask' + '_' + str(i).zfill(2) + '.png')
#     #     im2 = Image.fromarray(diff_mask)
#     #     #im3 = Image.fromarray(x[i].reshape(256,256).astype(np.uint8))
#     #     im4 = Image.fromarray(y[i].reshape(256,256).astype(np.uint8)*255)
#     #     im.save(pred_path + 'pred_mask' + '_' + str(i).zfill(2) + '.png')
#     #     im2.save(pred_path + 'diff_mask' + '_' + str(i).zfill(2) + '.png')
#     #     #im3.save(pred_path + 'test_img' + '_' + str(i).zfill(2) + '.png')
#     #     im4.save(pred_path + 'true_mask' + '_' + str(i).zfill(2) + '.png')



#     #earlystopping3 = EarlyStopping(monitor = 'TN', baseline = 0.01, patience = 15)

#     callbacks_list = [earlystopping1, earlystopping2, PredictionCallback()]


#     count = 0
#     for c in train_gen:
#         y = np.array(c[-1])
#         x = []
#         for j in range(len(c) - 1):
#             x.append(c[j])
#         x = np.array(x)

#         x = np.swapaxes(x,0,1)
#         x = np.swapaxes(x,1,2)
#         x = np.swapaxes(x,2,3)
#         if np.shape(x)[3] == 1:
#             x = np.squeeze(x, axis = 3)
#         if elast_deform == True:
#             for j in range(np.shape(x)[0]):
#                 if random.random() < prop_elastic:
#                     randoint = random.randint(0, 1e3)
#                     for k in range(channels):
#                         seed = np.random.RandomState(randoint)
#                         img = x[j,:,:,k]
#                         im_merge_t = elastic_transform(img, img.shape[1] * elast_alpha, img.shape[1] * elast_sigma, img.shape[1] * elast_affine_alpha, random_state = seed)

#                         im_t = im_merge_t[...,0]
#                         x[j,:,:,k] = im_t
#                     mask = y[j].copy().reshape(256,256)
#                     seed = np.random.RandomState(randoint)
#                     im_merge_t = elastic_transform(mask, mask.shape[1] * elast_alpha, mask.shape[1] * elast_sigma, mask.shape[1] * elast_affine_alpha, random_state = seed)

#                     im_mask_t = im_merge_t[...,0]
#                     y[j] = im_mask_t.reshape(256,256,1)
#         y = np.around(y / 255.)

#         results = m.fit(x, y, verbose = 0, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
#         i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
#         i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
#         i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
#         i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
#         i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
#         i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

#         count += 1
#         print(max_count - count)
#         if count >= max_count:
#             break
#     with open('results/{}/results.txt'.format(gpu), 'a') as f:
#         f.write('\n')
#         for key, value in i.items():
#             if list(i.keys())[-1] == key:
#                 f.write('%s' % value)
#             else:
#                 f.write('%s\t' % value)
