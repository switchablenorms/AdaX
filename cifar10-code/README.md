# The CIFAR-10 Training Code
<p align="center"><img width="100%" src="pics/CIFAR_IN.png"/></p>

## Description

This directory contains codes we use to train ResNet-20 on CIFAR-10. The default hyper-parameters are in [cfgs/cifar10/AdaXWcfg_res20.yaml](cfgs/cifar10/). Specifically for AdaX, we use 

+ `lr=5e-3, weight_decay=5e-2, betas=(0.9,1e-4)`

We multiply the learning rate by 0.1 at the 100th and the 150th epoch.

## Usage

+ Please download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset 
+ Please change the root (both `train_root` and `val_root`) of the dataset in [cfgs/cifar10/AdaXWcfg_res20.yaml](cfgs/cifar10/) to your CIFAR-10 root before running the code.
+ Please install [TensorboardX](https://pypi.org/project/tensorboardX/) and [YAML](https://pypi.org/project/PyYAML/)

Please use the following commands to run the code

+ `save_path=YOUR_SAVE_PATH`
+ `python -u Adax_main_cifar.py --save_path ${save_path} --config cfgs/cifar10/AdaXWcfg_res20.yaml --dataset cifar10`


To see the training and evaluation progress, use the following command

+ `tensorboard --logdir=YOUR_SAVE_PATH`
