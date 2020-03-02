# FIBSEM
Deep learning for segmentation of FIB-Sem volumetric image data

## Preprocessing
- ZCA whitening
- <s> Intensity gradient </s>(hard to see if it actually exists)
- <s>PCA </s> (Use ZCA instead)
- <s> Mapp intensities to either 0 or 1 </s>

## Data augmentation
- [ ] Rotation
- [ ] Flip
- [ ] Zoom
- [ ] Elastic deformation

## Architectures
- [ ] Unet
    - How does the upsampling part increase the resolution in the output? Inverse convolution operation. Can be achived in multiple ways.
- [ ] Simple as possible

## Optimisers
- [ ] Adam
- [ ] Which loss function to use?
- [ ] Metric? accuracy? Jaccard? dice-coefficient?

## Pre-trained weights

## Traning, validation and test split
- [ ] 60/20/20
