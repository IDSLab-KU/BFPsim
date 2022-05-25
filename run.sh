echo "=============================================="

# bfp Training
# docker run --rm --gpus '"device=4"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR10 --log True --bfp ResNet18_FB24


# Learning rate control of FB12_LG24
# docker run --rm --gpus '"device=4"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB12_LG24

# Dynamic Optimizer
# docker run --rm --gpus '"device=5"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB12_LG24 --do Simple/0.1/0.2/150/1/L8 --do-color False


# Test execution (easier to manage log folders)
# docker run --rm --gpus '"device=0,1"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_test --do Simple/0.4/0.5/0/1 

# CIFAR
# docker run --rm --gpus '"device=3"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR10 --log True --bfp ResNet18_test --do Simple/0.6/0.8/0/1 --do-color False


# Train Baselines
# docker run --rm --gpus '"device=2,3"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18
# docker run --rm --gpus '"device=5"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True 