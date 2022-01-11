

docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True --bfp ResNet18_FB24

# docker run --rm --gpus '"device=0,1,2,3"' --cpus="32" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:main python3 -u /app/imagenet.py --arch resnet18
