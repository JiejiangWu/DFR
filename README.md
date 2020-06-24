# Differentiable Function Rendering

This repository contains the code of the paper "DFR: Differentiable Function Rendering for Learning 3D Generation from images". 

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

Create an anaconda environment called `dfr` using
```
conda env create -f dfr.yaml
conda activate dfr
```

Then, compile the extension modules.
```
python setup.py build_ext --inplace
```

## Demo

We provide four demos as illustrated in our paper.

### 1. Test runtime for rendering an implicit function.

The detailed definition of network can be seen in `dfr/models.py`. 
We load the pre-trained implicit function from `test/checkpoints/gan-chair.pth.tar`
```
python test1_run_time.py
```
This script will render the implicit function from different views and save them as `test1_x.png`.
The runtime will be printed in console.

### 2. Test differentiability
Given a renference image and a neural-network defined function, you can optimize the function to fit the renference image.

We provide an example image with resolution 224x224.

You can try this optimization process with other images, just replace the `./input/input_test2.png` and run
```
python test2_differentiable.py
```
The script will create a gif `./test2.gif` and the optimized mesh `./test2.off` to illustrate the process 

## test pretrained model
We provide the pretrained model of single-image 3D reconstruction and image-based 3D GAN.
### 1. single-image 3D reconstruction
You can run the single-image 3D reconstruction via
```
test3_reconstruct.py
```

This script will read images in `./reconstruction/input` and save reconstructed shapes in `./reconstruction/output`.
### 2. 3D GAN
You can generate random shapes via
```
python test4_gan.py
```

This script will randomly sample noise vectors and generate 3D shapes from them. The results are saved in `./gan`