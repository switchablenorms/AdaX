#The CIFAR-10 Training Code


## Usage

+ Please change the root of the dataset in [cfgs/cifar10/AdaXWcfg_res20.yaml](cfgs/cifar10/) to your CIFAR-10 root before running the code.
+ Please install [TensorboardX](https://pypi.org/project/tensorboardX/) and [YAML](https://pypi.org/project/PyYAML/)

Please use the following command to run the code

+' save_path=YOUR_SAVE_PATH
+`python -u Adax_main_cifar.py --save_path ${save_path} --config cfgs/cifar10/AdaXWcfg_res20.yaml --dataset cifar10'