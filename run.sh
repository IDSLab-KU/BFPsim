


# Execute using docker
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/_test.py

# Running Simplenet
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_16 --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_8 --stat True
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_4 --stat True

# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m ResNet18 -bf ResNet18_4_OutX --stat True --save True


# Running zero-test simulation
docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app 4bd4764d9367 python3 -u /app/main.py --save-file ResNet18.model --mode zero-test --zt-bf False # --log False
