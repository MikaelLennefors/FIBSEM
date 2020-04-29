import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from GPyOpt.methods import BayesianOptimization
from PIL import Image

from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.optimizers import Adam

sys.path.insert(1, './networks')
import losses
from unet import unet
from dunet import D_Unet
from multiresunet import MultiResUnet
from nestnet import Nest_Net, bce_dice_loss

sys.path.insert(1, './processing')
from elastic_deformation import elastic_transform
from split_data import gen_data_split
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

if int(os.environ["CUDA_VISIBLE_DEVICES"]) == 0:
   gpu = 'Xp'

data_path = './data/train_val_data_border_clean/'
test_path = './data/test_data_border_clean/'
log_path = './results/{}/log.out'.format(gpu)
weights_path = './results/{}/weights'.format(gpu)
pred_path = './results/{}/masks/'.format(gpu)
callback_path = './results/{}/callback_masks/'.format(gpu)

grid_split = 0
grid_split = 2**grid_split


NO_OF_EPOCHS = 1
max_count = 1

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
            # horizontal_flip = True,
            #shear_range = 0.01,
            #rotation_range = 20,
            #zoom_range = 0.01,
            fill_mode = 'reflect'
        )

whitening = True
zca_coeff = 1e-2

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

# images = images[:,1:257,1:257,:]
#
# test_img = test_img[:,1:257,1:257,:]




if grid_split > 1:
    images, masks = split_grid(images, masks, grid_split)
    test_img, test_mask = split_grid(test_img, test_mask, grid_split, test_set = True)

test_img_zca = zca_whitening(test_img, zca_coeff)
test_img = test_img / 255.

test_mask = test_mask / 255.

t_gen = []
t_gen_zca = []
v_img = []
v_img_zca = []
v_mask = []
bins = 2

resampling_const = 2

k_fold = 1

for i in range(k_fold):
    print("nu går vi in i gen_data_split nr", i+1)
    train_images, train_mask, val_images, val_mask = gen_data_split(images, masks)
    if grid_split > 1:
        x = np.mean(train_mask, axis = (1,2,-1))/255
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
        train_mask = train_mask[new_indices]
    if channels > 1:
        train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], channels, 1)
    else:
        train_images = train_images.reshape(-1, np.shape(train_images)[1], np.shape(train_images)[2], 1)
    train_mask = train_mask.reshape(-1, np.shape(train_mask)[1], np.shape(train_mask)[2], 1)
    print(np.shape(train_images))
    print(np.shape(train_mask))


    train_images_zca = zca_whitening(train_images, zca_coeff)
    val_images_zca = zca_whitening(val_images, zca_coeff)

    train_images = train_images / 255.
    val_images = val_images / 255.

    aug_batch = np.shape(train_images)[0]
    train_gen = gen_aug(train_images, train_mask, aug_args, aug_batch)
    train_gen_zca = gen_aug(train_images_zca, train_mask, aug_args, aug_batch)

    t_gen.append(train_gen)
    t_gen_zca.append(train_gen_zca)
    v_img.append(val_images)
    v_img_zca.append(val_images_zca)
    v_mask.append(val_mask)
result_dict = []
def evaluate_network(parameters):
    parameters = parameters[0]
    # print(parameters)
    net_drop = parameters[0]
    net_filters = int(parameters[1])
    net_lr = parameters[2]
    prop_elastic = parameters[3]
    b_size = int(parameters[4])
    whitening = bool(parameters[5])
    zero_weight = np.mean(train_mask) / 255.
    mean_benchmark = []
    net_lr = math.pow(10,-net_lr)
    print('drop:', net_drop, '\nfilters:', net_filters, '\nlr:', net_lr, '\nprop el:', prop_elastic, '\nb_size:', b_size, '\nwhitening:', whitening)
    for i in range(k_fold):
        if whitening == True:
            train_gen = t_gen_zca[i]
            val_img = v_img_zca[i]
        else:
            train_gen = t_gen[i]
            val_img = v_img[i]
        val_mask = v_mask[i]

        if channels == 1:
            input_size = np.shape(val_img)[1]
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        else:
            input_size = (np.shape(val_img)[1],channels)
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        m.compile(optimizer = Adam(lr = net_lr), loss = bce_dice_loss, metrics = [losses.iou_coef, 'accuracy'])
        # m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy'])



        earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = 100, mode = 'max')
        earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.6, patience = 50)
        class PredictionCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
                # new_shape = int(np.shape(predic_mask)/gridsplit)
                testy_mask = np.around(predic_mask).reshape(np.shape(predic_mask)[1],np.shape(predic_mask)[1]).astype(np.uint8)*255
                print('')
                print("True proportion:", np.mean(test_mask[0]), "Predicted proportion:", np.mean(testy_mask)/255)
                print('')
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
            max_val = max(np.max(x), np.max(y), np.max(val_img), np.max(val_mask))
            if max_val > 1:
                raise "max_value_error, kolla zca"
            try:
                results = m.fit(x, y, verbose = 1, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            except KeyboardInterrupt:
                import tabulate
                print('\n-')
                max_lines = int(input("Input the number of top results to print: "))
                print('-')
                max_lines = min(len(result_dict), max_lines)
                header = result_dict[0].keys()
                rows =  [x.values() for x in sorted(result_dict, key = lambda m: m['Mean\nIoU'],reverse=True)[:max_lines]]
                print('\n\n')
                print(tabulate.tabulate(rows, header, tablefmt="fancy_grid", stralign="right", floatfmt=(".3f", "2d", ".2e", "s", "d", ".3f", ".3f")))
                print('\n\n')
                sys.exit()
            count += 1
            if count >= max_count:
                break
        if whitening == True:
            pred = m.evaluate(test_img_zca, test_mask, verbose = 2)
        else:
            pred = m.evaluate(test_img, test_mask, verbose = 2)
        score = pred[1]

        mean_benchmark.append(score)
    m1 = np.mean(mean_benchmark)
    result_dict.append({'Mean\nIoU': m1,
                    'Filters': net_filters,
                    'Learning\nrate': net_lr,
                    'Whitening': whitening,
                    'Batch\nsize': b_size,
                    'Dropout': net_drop,
                    'Elastic\nproportion': prop_elastic})
    return m1


bds = [{'name': 'net_drop', 'type': 'continuous', 'domain': (0.3, 0.5)},
        {'name': 'net_filters', 'type': 'discrete', 'domain': (16, 32, 64)},
        {'name': 'net_lr', 'type': 'continuous', 'domain': (3, 6)},
        {'name': 'prop_elastic', 'type': 'continuous', 'domain': (0, 0.2)},
        {'name': 'b_size', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 6)},
        {'name': 'whitening', 'type': 'discrete', 'domain': (0, 1)}]

pbounds = {'net_drop': (0.49,0.5),
    'net_filters': (5.0, 6.0),
    'net_lr': (3.9, 4.1),
    'prop_elastic': (0.0, 0.2)
    }

optimizer = BayesianOptimization(f=evaluate_network,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)


# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=100)

optimizer.plot_acquisition()
optimizer.plot_convergence()
# optimizer = BayesianOptimization(
#     f=evaluate_network,
#     pbounds=pbounds,
#     verbose=2,
# )
#
# start_time = time.time()
# optimizer.maximize(init_points=10, n_iter=100,)
# time_took = time.time() - start_time
#
# # print(f"Total runtime: {hms_string(time_took)}")
# print(optimizer.max)
