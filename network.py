import numpy as np
import os
import param
import random
import sys
import tensorflow as tf
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = input("Choose GPU, 0 for Xp, 1 for V: ")

while int(os.environ["CUDA_VISIBLE_DEVICES"]) not in [0,1]:
    os.environ["CUDA_VISIBLE_DEVICES"] = input("Choose GPU, 0 for Xp, 1 for V: ")

channels = int(input("Choose channels, 1, 3, 5 or 7: "))
while channels not in [1,3,5,7]:
    channels = int(input("Choose channels, 1, 3, 5 or 7: "))

gpu = 'V'
if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
    gpu = 'Xp'

#if channels > 1:
data_path = './data/train_val_data_border/'
test_path = './data/test_data_border/'
#else:
#    data_path = './data/train_val_data/'
#    test_path = './data/test_data/'
log_path = './results/{}/log.out'.format(gpu)
weights_path = './results/{}/weights'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

params = param.HyperParam()

configurations = [params.generate_hyperparameters() for _ in range(10000)]

grid_split = 0


NO_OF_EPOCHS = 50
aug_batch = 180
max_count = 3
b_size = 1
elast_deform = True
elast_alpha = 2
elast_sigma = 0.08
elast_affine_alpha = 0.08
net_filters = 32
prop_elastic = 0.05
net_lr = 1e-4
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

train_gen, val_img, val_mask, test_img, test_mask = gen_data_split(path_to_data = data_path, path_to_test = test_path,
        channels = channels, b_size = aug_batch, whitening_coeff = zca_coeff, maskgen_args = aug_args, grid_split = grid_split)

for i in configurations:


    # net_filters = int(i['net_filters'])
    # prop_elastic = i['prop_elastic']
    # # net_activ_fun = int(i['net_activ_fun'])
    # net_lr = i['net_lr']
    # # net_bin_split = i['net_bin_split']
    # net_drop = i['net_drop']
    #zca_coeff = i['zca_coeff']
    i['val_iou'] = 0
    i['val_accuracy'] = 0
    i['val_TP'] = 0
    i['val_TN'] = 0
    i['val_FP'] = 0
    i['val_FN'] = 0

    # conf_name = 'logs/{}_{}_{:.2e}_{:.2f}_{}_{:.2f}_{:.2f}_{}'.format(channels, net_filters, net_lr, net_bin_split, net_activ_fun,
    #     net_drop, prop_elastic, datetime.now().strftime('%m-%d_%H:%M:%S'))
    #print(np.shape(val_img))
    #raise
    if channels == 1:
        input_size = np.shape(val_img)[1]//(2**grid_split)
        m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        #m = Nest_Net(input_size, color_type=1, num_class=1, deep_supervision=False)
        #m = MultiResUnet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, bin_weight = net_bin_split, dout = net_drop)
    else:
        input_size = (np.shape(val_img)[1]//(2**grid_split),channels)
        m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
    m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy', losses.TP, losses.TN, losses.FP, losses.FN])
    #checkpoint = ModelCheckpoint(weights_path, monitor='val_iou_coef',
    #                             verbose=1, save_best_only=True, mode='max')

    #csv_logger = CSVLogger(log_path, append=True, separator=';')

    #tensorboard = TensorBoard(log_dir = conf_name)

    earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.001, patience = 100, mode = 'max')
    earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.5, patience = 30, mode = 'max')
    class PredictionCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
            testy_mask = np.around(predic_mask).reshape(256,256).astype(np.uint8)*255

            #diff_mask = np.abs(testy_mask - y[i].reshape(256,256)).astype(np.uint8)*255
            im = Image.fromarray(testy_mask)
            im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')
            #im2.save(pred_path + 'diff_mask' + '_' + str(i).zfill(2) + '.png')
    #     im2 = Image.fromarray(diff_mask)
    #     #im3 = Image.fromarray(x[i].reshape(256,256).astype(np.uint8))
    #     im4 = Image.fromarray(y[i].reshape(256,256).astype(np.uint8)*255)
    #     im.save(pred_path + 'pred_mask' + '_' + str(i).zfill(2) + '.png')
    #     im2.save(pred_path + 'diff_mask' + '_' + str(i).zfill(2) + '.png')
    #     #im3.save(pred_path + 'test_img' + '_' + str(i).zfill(2) + '.png')
    #     im4.save(pred_path + 'true_mask' + '_' + str(i).zfill(2) + '.png')



    #earlystopping3 = EarlyStopping(monitor = 'TN', baseline = 0.01, patience = 15)

    callbacks_list = [earlystopping1, earlystopping2, PredictionCallback()]


    count = 0

    if channels == 7:
        for x1, x2, x3, x4, x5, x6, x7, y in train_gen:
            x = np.stack((x1,x2,x3,x4,x5,x6,x7))
            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)

            if elast_deform == True:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        mask = y[j].copy()
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge = np.concatenate((img, mask), axis=2)
                            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * elast_alpha, im_merge.shape[1] * elast_sigma, im_merge.shape[1] * elast_affine_alpha, random_state = seed)

                            im_t = im_merge_t[...,0]
                            im_mask_t = im_merge_t[...,1]
                            x[j,:,:,k] = im_t
                            y[j] = im_mask_t
            y = np.around(y / 255.)

            results = m.fit(x, y, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

            count += 1
            print(max_count - count)
            if count >= max_count:
                break
    elif channels == 5:
        for x1, x2, x3, x4, x5, y in train_gen:
            x = np.stack((x1,x2,x3,x4,x5))
            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)

            if elast_deform == True:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        mask = y[j].copy()
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge = np.concatenate((img, mask), axis=2)
                            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * elast_alpha, im_merge.shape[1] * elast_sigma, im_merge.shape[1] * elast_affine_alpha, random_state = seed)

                            im_t = im_merge_t[...,0]
                            im_mask_t = im_merge_t[...,1]
                            x[j,:,:,k] = im_t
                            y[j] = im_mask_t
            y = np.around(y / 255.)

            results = m.fit(x, y, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

            count += 1
            print(max_count - count)
            if count >= max_count:
                break
    elif channels == 3:
        for x1, x2, x3, y in train_gen:
            x = np.stack((x1,x2,x3))
            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)

            if elast_deform == True:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        mask = y[j].copy()
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge = np.concatenate((img, mask), axis=2)
                            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * elast_alpha, im_merge.shape[1] * elast_sigma, im_merge.shape[1] * elast_affine_alpha, random_state = seed)

                            im_t = im_merge_t[...,0]
                            im_mask_t = im_merge_t[...,1]
                            x[j,:,:,k] = im_t
                            y[j] = im_mask_t
            y = np.around(y / 255.)

            results = m.fit(x, y, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

            count += 1
            print(max_count - count)
            if count >= max_count:
                break
    else:
        for x, y in train_gen:
            x = np.array(x)

            # if elast_deform == True:

            #     for j in range(np.shape(x)[0]):
            #         if random.random() < prop_elastic:
            #             img = x[j]
            #             mask = y[j]
            #             im_merge = np.concatenate((img, mask), axis=2)
            #             im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * elast_alpha, im_merge.shape[1] * elast_sigma, im_merge.shape[1] * elast_affine_alpha)

            #             im_t = im_merge_t[...,0]
            #             im_mask_t = im_merge_t[...,1]
            #             x[j] = im_t
            #             y[j] = im_mask_t
            y = np.around(y / 255.)
            results = m.fit(x, y, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))
            count += 1
            print(max_count - count)
            if count >= max_count:
                break
    with open('results/{}/results.txt'.format(gpu), 'a') as f:
        f.write('\n')
        for key, value in i.items():
            if list(i.keys())[-1] == key:
                f.write('%s' % value)
            else:
                f.write('%s\t' % value)

    # m.evaluate(x = test_img, y = test_mask)
    # # new_mask = np.around(m.predict(np.expand_dims(val_img[0], axis = 0))).reshape(256,256,1)


    # for i in range(len(x)):
    #     testy_mask = m.predict(np.expand_dims(x[i], axis = 0)).reshape(256,256).astype(np.uint8)*255
    #     #print(np.max(testy_mask))
    #     diff_mask = np.abs(testy_mask - y[i].reshape(256,256)).astype(np.uint8)*255
    #     im = Image.fromarray(testy_mask)
    #     im2 = Image.fromarray(diff_mask)
    #     #im3 = Image.fromarray(x[i].reshape(256,256).astype(np.uint8))
    #     im4 = Image.fromarray(y[i].reshape(256,256).astype(np.uint8)*255)
    #     im.save(pred_path + 'pred_mask' + '_' + str(i).zfill(2) + '.png')
    #     im2.save(pred_path + 'diff_mask' + '_' + str(i).zfill(2) + '.png')
    #     #im3.save(pred_path + 'test_img' + '_' + str(i).zfill(2) + '.png')
    #     im4.save(pred_path + 'true_mask' + '_' + str(i).zfill(2) + '.png')

    # m.save('Model.h5')
