

# Execute example train config file
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -tc example

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -bf ResNet18_FB12_B --loss-boost 2

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train -tc MLPMixerB16_ImageNet_FB12
conf=MLPMixerB16_ImageNet
echo ${conf}
# Imagenet
docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --shm-size 24G --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc ${conf}

# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode train --log True --stat True -tc MLPMixerB16_ImageNet_FB12

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --mount type=bind,source=/dataset,target=/dataset --workdir /app -v "$(pwd)":/app $(whoami)/floatblock:latest python3 -u /app/main.py --mode generate-config --log False