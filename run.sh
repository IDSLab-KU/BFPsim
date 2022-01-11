
docker run --rm --gpus '"device=0,1,2,3"' --cpus="32" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:main python3 -u /app/imagenet.py --arch resnet18


# docker run --rm --gpus '"device=4,5,6,7"' --cpus="32" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:main python3 -u /app/imagenet.py --arch mobilenet_v2 --bfp MobileNetv2_FB24
