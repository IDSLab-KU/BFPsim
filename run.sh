

# Execute example train config file
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -tc example

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -bf ResNet18_FB12_B --loss-boost 2

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train -tc MLPMixerB16_ImageNet_FB12

# Imagenet
# conf=ResNet18_ImageNet
# echo ${conf}
# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc ${conf}

# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc MLPMixerB16_ImageNet_FB12

# generate-config
# docker run -ti --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode generate-config --log False --model ResNet18 --dataset ImageNet --dataset-path /dataset/ImageNet

# zse-analyze
# docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./Results/saves/ResNet18_CIFAR10_8Lbit_finish.model --bf-layer-conf-file ResNet18_FB16_B --zse-graph-mode none

mode=FB12
docker run --rm --gpus '"device=1"' --cpus="8" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode zse-analyze --log False --num-workers 2 --model ResNet18 --save-file ./trained.model --bf-layer-conf-file ResNet18_${mode} --zse-graph-mode none --log True --save-name ZSE_ResNet18_${mode}