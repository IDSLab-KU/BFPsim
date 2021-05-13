# Temp exection file for debugging
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m AlexNet -bf AlexNet_4 --log False
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m AlexNet --log False --save True --stat True
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_4 --log False --print-train-count 100
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m VGG16 -bf VGG16_8

# zero-test mode
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode zero-test --save-file ./saves/ResNet18_CIFAR10_FP32_finish.model --log False

# data-save mode
# docker run --rm --gpus '"device=1"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode save-data --save-file ./logs/BatchNormAnalysis/20210413_065950_010.model --model ResNet18


# Execute numba test
# docker run --rm --gpus '"device=2"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m AlexNet -bf AlexNet_16 --dataset CIFAR10
# docker run --rm --gpus '"device=2"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ResNet18 -bf ResNet18_16 --dataset CIFAR10

docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode train -tc AlexNet_00 --log True --stat True

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py --mode temp -m ResNet18 -bf ResNet18_4 --log False 

# Execute
# MODEL=DenseNetCifar
MODEL=MobileNetv1
DATASET=CIFAR10
BIT=4L

CPUS=4

# Save with savefile name
# docker run --rm --gpus '"device=0"' --cpus="${CPUS}" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_FP32
# docker run --rm --gpus '"device=3"' --cpus="${CPUS}" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit

# for MODEL in AlexNet ResNet18 DenseNetCifar MobileNetv1 VGG16
# do
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_FP32
# for BIT in 4 8 16
# do
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# done
# done
