# FIBSEM
Deep learning for segmentation of FIB-Sem volumetric image data

## Preprocessing
- ZCA whitening
- <s> Intensity gradient </s>(hard to see if it actually exists)
- <s>PCA </s> (Use ZCA instead)
- <s> Mapp intensities to either 0 or 1 </s> (if that would have workedm we would not need at network)

## Data augmentation
- Rotation
- Flip (vertical and horizontal)
- Zoom
- Shear
- Elastic deformation

## Architectures
- Unet (baseline)
- MultiResUNet
- 3D UNet
- D-UNet
- Non-local UNet

## Network blocks
- Batch Normilization or not
- Different kinds of up sampling:
  - UP-sampling (baseline)
  - Inverse convolution (deconvolution)
  - VoxelDCL

## Optimisers
- Adam

## Loss function
- Binary Cross Entropy
- IoU
- Dice
- Mixed

## Metric
- Jaccard/IoU
- Dice-coefficient
- Accuracy (ust for comparison to old results with random forrest)

## Pre-trained weights

## Traning, validation and test split
- 60/20/20
