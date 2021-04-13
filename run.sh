docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet --save True --stat True

# Etc
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/_testbench.py
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet -bf AlexNet_4
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 -bf ResNet18_8
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m AlexNet

# Execute using docker
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/_test.py

# Running Simplenet
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m SimpleNet --stat True --save True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m SimpleNet -bf SimpleNet_16 --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_8 --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_4 --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m ResNet18 -bf ResNet18_4_OutX --stat True --save True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 --stat True --save True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 -bf ResNet18_8_54_X --stat True --save True

# ResNet18 with others
# docker run --rm --cpus="4" --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 --stat True --save True --training-epochs 50
# docker run --rm --cpus="4" --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 --bf ResNet18_8_54_X --stat True --save True --training-epochs 50
# docker run --rm --cpus="4" --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py -m ResNet18 --bf ResNet18_16_54_X --stat True --save True --training-epochs 50

# Running zero-test simulation
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py --save-file ResNet18.model --mode zero-test --zt-bf False # --log False
