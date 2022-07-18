# Semi-Leak

This is a PyTorch implementation of Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning, as described in our paper:

Xinlei He, Hongbin Liu, Neil Zhenqiang Gong, Yang Zhang: Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning (ECCV 2022)

Our code is based on `Python 3.8.12` and `Torch 1.12.0`.

Regarding the implementation of SSL methods, we refer to code on [TorchSSL](https://github.com/TorchSSL/TorchSSL).

## Step 0: Generate config files
```
cd scripts &&
python config_generator.py &&
cd ..  

```

## Step 1: Train targegt/shadow models
```
python fixmatch.py --c config/fixmatch/fixmatch_cifar10_500_0.yaml --mode target
python fixmatch.py --c config/fixmatch/fixmatch_cifar10_500_0.yaml --mode shadow

python uda.py --c config/uda/uda_cifar10_500_0.yaml --mode target
python uda.py --c config/uda/uda_cifar10_500_0.yaml --mode shadow

python flexmatch.py --c config/flexmatch/flexmatch_cifar10_500_0.yaml --mode target
python flexmatch.py --c config/flexmatch/flexmatch_cifar10_500_0.yaml --mode shadow
```

## Step2: Query targegt/shadow models
```
python query_target.py --dataset cifar10 --num_classes 10 --num_labels 500 --widen_factor 2 --target_epoch 100 --save_dir ./saved_models 
```

## Step 3: Perform membership inference attacks
###  Conventional attacks
```
# NN-based attacks
python mia_normal.py --attack_name black-box --ssl_method fixmatch --dataset cifar10 --num_classes 10 --num_labels 500 --target_epoch 100 --save_dir ./saved_models

# Metric-based attacks
python mia_normal.py --attack_name metric --ssl_method fixmatch --dataset cifar10 --num_classes 10 --num_labels 500 --target_epoch 100 --save_dir ./saved_models

```

### Data augmentation-based attacks
```

python mia_augmented.py --ssl_method fixmatch  --dataset cifar10 --num_classes 10 --num_labels 500 --target_epoch 100 --save_dir ./saved_models --similarity_func jensenshannon --augmented_num 10

```
