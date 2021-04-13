# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_4 --log False

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_4 --log False --print-train-count 100


# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 -bf VGG16_8




# Execute
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet --save True --stat True
docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_4 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_8 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_16 --save True --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 -bf ResNet18_4 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 -bf ResNet18_8 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 -bf ResNet18_16 --save True --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 -bf VGG16_4 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 -bf VGG16_8 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 -bf VGG16_16 --save True --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m MobileNetv1 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m MobileNetv1 -bf MobileNetv1_4 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m MobileNetv1 -bf MobileNetv1_8 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m MobileNetv1 -bf MobileNetv1_16 --save True --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_4 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_8 --save True --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_16 --save True --stat True
