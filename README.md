


# BFPSim


# Features
This repository features...

- Training various neural networks with [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
- Fully configurable training environment
- Save checkpoints, logs, etc

# Installation

## Setup with docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/) on the targeted machine.
2. Clone this repository with `git clone https://github.com/IDS-Lab-DGIST/BFPsim`
3. Run Installation `./install.sh` (You may need to edit file's authority using chmod)


## Setup tensorboard

After creating docker container, you should execute tensorboard on background.

On this repository, just execute `./tensorboard.sh [External Port]`. External Port can set manually, take care that it doesn't conflict with other user if you are using remote server. Recommended value is `6006`.

If your running this on remote server, make sure you opened the external port using `ufw` or else, then input `[Remote Server IP]:[External Port]` on your favorite internet browser.

If you are running this on local machine, just type `http://localhost:[External Port]`.


## Setup without docker
1. Clone this repository
2. Install requirements listed below 
- `torch >= 1.9.1`
- `torchvision >= 0.5.0`
- `numba >= 0.53.1`
- `matplotlib >= 3.4.2`
- `einops >= 0.3.0`
- `slack_sdk`
- `tensorboard >= 2.7.0` (tensorboard can be any version, though)

# Execution examples

## Resnet with preset config

For the simple execution of the ResNet18 with the FlexBlock12 data structure, execute

```docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:main python3 -u /app/main.py --model --arch resnet18```

## More information

More specifically, look at the [docs](/docs/_index.md) for the arguments and setup your custom network, etc...

# Citation

# License

This repository uses [CC BY 4.0](https://creativecommons.org/licenses/)