# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_4 --log False

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m DenseNetCifar -bf DenseNetCifar_4 --log False --print-train-count 100


# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m VGG16 -bf VGG16_8




# Execute



# MODEL=AlexNet
DATASET=CIFAR10

for MODEL in AlexNet ResNet18 DenseNetCifar MobileNetv1 VGG16
do
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ${MODEL} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_FP32
# for BIT in 4 8 16
# do
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ${MODEL} -bf ${MODEL}_${BIT} --dataset ${DATASET} --save True --stat True --save-name ${MODEL}_${DATASET}_${BIT}bit
# done
done