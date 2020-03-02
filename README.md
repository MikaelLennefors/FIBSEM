# FIBSEM
Deep learning for segmentation of FIB-Sem volumetric image data

## Preprocessing
- [ ] Intensity gradient (NO: hard to see if it actually exists)
- [x] ZCA whitening
- <s>PCA </s>
- [ ] Mapp intensities to either 0 or 1

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
