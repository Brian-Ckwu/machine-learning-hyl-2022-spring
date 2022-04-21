# ML-2022-Spring Homework 6

To reproduce the results on JudgeBoi:

## Download Data

Download the data (faces.zip) and unzip it. The 1000 images should be in the `./faces` directory.

## Install Simple StyleGan2 for Pytorch

    conda install pytorch torchvision -c python

## Train the Model

    stylegan2_pytorch --data ./faces --name GAN-64 --image-size 64 --attn-layers [1, 2]

The training process should take about 12-24 hrs.

## Generate the Images

    bash ./generate_images.sh

The images would be stored in `./preds/GAN-64`.