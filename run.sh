echo "Nothing here..."

# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True
# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR10 --log True --bfp ResNet18_FB12 --do Simple/0.35/0.5/0/1

# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True

# docker run --rm --gpus '"device=5"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True --training-epochs 300

# docker run --rm --gpus '"device=4"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR10 --log True # --bfp ResNet18_FB24

# docker run --rm --gpus '"device=4"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True --bfp ResNet18_FB24

# docker run --rm --gpus '"device=4,5"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB16
# docker run --rm --gpus '"device=0,1"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18 --bfp ResNet18_FB24 --do Simple/0.35/0.5/0/1

# docker run --rm --gpus '"device=2,3"' --cpus="64" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/imagenet.py --arch resnet18
