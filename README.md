# BFPSim
This repository can simulates block-floating point arithmetic in software manner within reasonable time cost (~2x time)

# Features
This repository features...

- Training various neural networks with [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
- Fully configurable training environment
- Fully configurable block floating point on precision and group size on forward, weight gradient, local gradient
- Train on ImageNet / CIFAR100
- Simple dynamic precison control
- Save checkpoints, logs, etc

# Installation

## Setup with docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/) on the targeted machine.
2. Clone this repository
3. Make docker image as `docker build . -t $(whoami)/bfpsim:latest` (It will take a while, so get some coffee break :coffee:).

## Setup tensorboard (NOT Recommended)

After creating docker container, you should execute tensorboard on background.

To execute tensorboard, move to `tensorboard` directory, and execute `./run.sh [External Port]`. Make sure you are in tensorboard directory. Recommended value for external port is `6006`, but you can change if you know what you are doing. It will take a while if you first executing tensorboard, since docker image is different from main image

If your running this on remote server, make sure you opened the external port using `ufw`, `iptables`, etc, then input `[Remote Server IP]:[External Port]` on your beloved internet browser.

If you are running this on local machine, just type `http://localhost:[External Port]`, and you're good to go. 😆


## Setup without docker
1. Clone this repository
2. Install requirements listed below 
- `torch >= 1.9.1`
- `torchvision >= 0.5.0`
- `numba >= 0.53.1`
- `matplotlib >= 3.4.2`
- `einops >= 0.3.0`
- `slack_sdk`
- `tensorboard >= 2.7.0` (tensorboard version is not crucial, though)

## Additional Setup - Slackbot
It's possible to set train information to your own slack server.

If you want to set by your own, follow [Tutorial](https://github.com/slackapi/python-slack-sdk/blob/main/tutorial/01-creating-the-slack-app.md) and get the slackbot token, and put the token in separate file named `./slackbot.token`, and give `--slackbot` option as true.

# Execution examples

## Resnet with preset config

Executing ResNet18 on CIFAR100, with FP32

> ```docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/cifar.py --mode train --model ResNet18 --dataset CIFAR100 --log True```

Executing ResNet18 on CIFAR100, with FP24

> ```docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/cifar.py --mode train --model ResNet18 --dataset CIFAR100 --log True --bfp ResNet18_FB24```

Executing ResNet18 on ImageNet([Original Code](https://github.com/pytorch/examples/tree/main/imagenet)), with FP24

> ```docker run --rm --gpus '"device=0"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB12LG24 --log True```
`

If you are using imagenet, I recommend to reduce training epoch option as `--epoch 60`, otherwise it will cost a quite time. Learning rate will be automatically adjusted based on full training epoch(Reduce every 1/3 training phase).

Changing `'"device=0"'` will be change gpu to run. It is possible to use several gpus like `'"device=0,1"'`, but I can't sure it will run properly.😂

Executing runs will automatically added to the folder names `./runs/`. Visualization is also available on the tensorboard, which is mentioned before.

## How to make configuration file
Making your own configuration file on `./conf_net` is not simple...

First, `"default":` defines the default configuration of BFP for any convolution or fully-connected layer.
```text
    "default":{
        "fw_bit":4,
        "fi_bit":4,
        "bwo_bit":4,
        "bwi_bit":4,
        "biw_bit":4,
        "bio_bit":4,
        "fw_dim":[1,24,3,3],
        "fi_dim":[1,24,3,3],
        "bwo_dim":[1,24,3,3],
        "bwi_dim":[1,24,3,3],
        "biw_dim":[1,24,3,3],
        "bio_dim":[1,24,3,3]
    },
```
Each notation will indicates...
 - `fw`: (True/False) Do BFP on weight on forward pass
 - `fi`: (True/False) Do BFP on input while forward pass
 - `fo`: (True/False) Do BFP on output while forward pass
 - `bwo`: (True/False) Do BFP on output gradient while calculating weight gradient
 - `bwi`: (True/False) Do BFP on input feature map while calculating weight gradient
 - `bwg`: (True/False) Do BFP on weight gradient
 - `bio`: (True/False) Do BFP on output gradient while calculating local gradient
 - `biw`: (True/False) Do BFP on weight while calculating local gradient
 - `big`: (True/False) Do BFP on local gradient
 - `fw_bit`: Bit length of mantissa (precision) on weight on forward pass
 - `fi_bit`: Bit length of mantissa (precision) on input while forward pass
 - `fo_bit`: Bit length of mantissa (precision) on output while forward pass
 - `bwo_bit`: Bit length of mantissa (precision) on output gradient while calculating weight gradient
 - `bwi_bit`: Bit length of mantissa (precision) on input feature map while calculating weight gradient
 - `bwg_bit`: Bit length of mantissa (precision) on weight gradient
 - `bio_bit`: Bit length of mantissa (precision) on output gradient while calculating local gradient
 - `biw_bit`: Bit length of mantissa (precision) on weight while calculating local gradient
 - `big_bit`: Bit length of mantissa (precision) on local gradient
 - `fw_dim`: Group dimension of mantissa (precision) on weight on forward pass
 - `fi_dim`: Group dimension of mantissa (precision) on input while forward pass
 - `fo_dim`: Group dimension of mantissa (precision) on output while forward pass
 - `bwo_dim`: Group dimension of mantissa (precision) on output gradient while calculating weight gradient
 - `bwi_dim`: Group dimension of mantissa (precision) on input feature map while calculating weight gradient
 - `bwg_dim`: Group dimension of mantissa (precision) on weight gradient
 - `bio_dim`: Group dimension of mantissa (precision) on output gradient while calculating local gradient
 - `biw_dim`: Group dimension of mantissa (precision) on weight while calculating local gradient
 - `big_dim`: Group dimension of mantissa (precision) on local gradient

Group dimension is provided as list like `[x,y,z,w]`. Example is shown below.
 - Group size of 8, direction of input channel : `[1,8,1,1]`
 - Group size of 9, group by kernel (weight) : `[1,1,3,3]`
 - Group size of 216, mentioned on FlexBlock : `[1,24,3,3]`

If the desired size is not suffcient, it will group restover to one group. For example, if the length of input channel is 53(which is weird though), and a user want to group with 8 each, it will make 7 groups, but last group will have 5, not 8 elements.😋

My manually type a network's name with argument `"type=default"`, it will not make to BFPConv2d, and do the normal convolution. Just make sure to input the name correctly, and put `net.` in front of the layer's name.
```text
    "net.conv1":{
        "type":"default"
    },
```

## Simple Dynamic Precision Control

By adding optional argument `--do`, it will execute the dynamic optimizer with simple method mentioned on original paper, FlexBlock.
> ```docker run --rm --gpus '"device=5"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB12LG24 --do Simple/0.1/0.2/50/1/5 --do-color False```

Execution of CIFAR100, with simple dynamic pricison control
> ```docker run --rm --gpus '"device=3"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/cifar.py --mode train --model ResNet18 --dataset CIFAR10 --log True --bfp ResNet18_FB24 --do Simple/0.6/0.8/0/1 --do-color False```

`Simple/0.6/0.8/0/1` Each indicates,
 - Simple : Method to control
 - 0.6 : Threshold of zero-setting error that decrease precision
 - 0.8 : Threshold of zero-setting error that increase precision
 - 0 : fix precision to provided file for 0 steps at start of training
 - 1 : fix precision for 1 step if precision is changed

If you enable `-do-color` option, console window will shine like rainbow :), and see the zero-setting error of each weights/weight gradients/local gradients. (Recommended to show this, I really put some work to looks like a hacker :sunglasses:)

## Train Config
Using the `-tc` option will use train configuration file on `./conf_train`. It will only work on `cifar.py`, and you can set various preset arguments to the file, so you don't have to input arguments manually.

Execution example is shown below.
> ```docker run --rm --gpus '"device=3"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/apwhoami)/bfpsim:latest python3 -u /app/cifar.py --mode train -tc ResNet18_CIFAR100_Mixed```

By writing `bfp-layer-conf-dict` and `optimizer-dict` like a python dict, you can manually set your arguments over training epochs. Make sure you have to input scheduler-step if you want to change model's precision, learning rate scheduler step need to be proceeded to match training status.
```text
    "bfp-layer-conf-dict":{
        "0":"ResNet18_FB16",
        "20":"ResNet18_FB12LG16",
        "180":"ResNet18_FB16"
    },
    "optimizer-dict":{
        "20":{
            "scheduler-step":20
        },
        "180":{
            "scheduler-step":180
        }
    }
```

In fact, code about setting training config is quite old code, I have no idea why this is working until now. 🤤

## Training on Transformer
Need to organize a bit... (Not prepared to open)

# Citation

[FlexBlock: A Flexible DNN Training Accelerator with Multi-Mode Block Floating Point Support](https://arxiv.org/abs/2203.06673)
