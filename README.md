


# FlexBlock

This repository contains source code execution of the FlexBlock's simulation.


# Features
This repository features...

- Training various neural networks with [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
- Fully configurable training environment
- Save checkpoints, logs, etc

# Installation

## Setup with docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/) on the targeted machine.
2. Clone this repository with `git clone https://github.com/r3coder/FlexBlock`
3. Make a docker container as: `docker build . -t $(whoami)/flexblock:latest`

## Setup without docker
1. Clone this repository
2. Install requirements listed below 
- `torch >= 1.7.1`
- `torchvision >= 0.5.0`
- `numba >= 0.50.1`
- `matplotlib >= 3.4.2`
- `einops >= 0.3.0`

# Execution examples

## Resnet with preset config

For the simple execution of the ResNet18 with the FlexBlock12 data structure, execute

```docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train --model ResNet18 -bf ResNet18_FB12```

It takes quite time to show the result (30min / epoch), so please be patient.

## More information

More specifically, look at the [docs](/docs/_index.md) for the arguments and setup your custom network, etc...

# Citation

# License

This repository uses [CC BY 4.0](https://creativecommons.org/licenses/)