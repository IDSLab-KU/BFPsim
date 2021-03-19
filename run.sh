


# Execute using docker
# docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/_test.py

docker run --rm --gpus '"device=0"' --workdir /app -v "$(pwd)":/app pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime python3 -u /app/main.py -m SimpleNet -bf SimpleNet_8
