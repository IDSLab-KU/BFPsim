
docker run --rm --gpus '"device=2"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/flexblock:latest python3 -u /app/__test.py

# Test
# docker run --rm --gpus '"device=2"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/flexblock:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log False

# AlexNetFB12
# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --model AlexNet --dataset CIFAR100 --bf-layer-conf-file AlexNet_FB12 --log False

# ResNet18
# docker run --rm --gpus '"device=2"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/flexblock:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --bf-layer-conf-file ResNet18_FB12 --log False

# docker run --rm --gpus '"device=3"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/flexblock:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log False

# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --bf-layer-conf-file ResNet18_FB12 --log False

# _test2
# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/_test2.py

# ImageNetTest
# docker run --rm --gpus '"device=3"' --cpus="16" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/_test.py --print-freq 100 --batch-size 128 ../dataset/ImageNet/Classification

# docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --bf-layer-conf-file ResNet18_FB12_B --log False

# Execute example train config file
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -tc example

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -bf ResNet18_FB12_B --loss-boost 2

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train -tc MLPMixerB16_ImageNet_FB12

# Imagenet
# conf=ResNet18_ImageNet
# echo ${conf}
# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc ${conf}

# docker run --rm --gpus '"device=all"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/test.py ../dataset/ImageNet/Classification --print-freq 50 --rank 4

# temp mode
# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode temp --log False --dataset ImageNet --dataset-path ../dataset/ImageNet/Classification

# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc MLPMixerB16_ImageNet_FB12

# generate-config
# docker run -ti --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode generate-config --log False --model ResNet18 --dataset ImageNet --dataset-path /dataset/ImageNet/Classification

# zse-analyze
# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./Results/saves/ResNet18_CIFAR10_8Lbit_finish.model --bf-layer-conf-file ResNet18_FB16_B --zse-graph-mode none

# Normal mode
# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./trained.model --bf-layer-conf-file ResNet18_ZSE_${mode} --zse-graph-mode none --log True --save-name ZSE_ResNet18_${mode}


# using original mode
# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./Results/saves/ResNet18_CIFAR10_4Lbit_finish.model --bf-layer-conf-file ResNet18_FB12 --zse-graph-mode none --save-name ZSE_ResNet18_FB12 --log True

# mode=8B
# epoch=200
# gs=108
# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./trained.model --bf-layer-conf-file ResNet18_ZSE_${mode}_${gs} --zse-graph-mode none --log False