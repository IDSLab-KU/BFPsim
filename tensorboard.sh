docker run --rm --user "$(id -u):$(id -g)" -p "$1":8088 -d --name=$(whoami)_bfpsim_tensorboard --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest tensorboard --logdir=runs --port 8088 --host 0.0.0.0