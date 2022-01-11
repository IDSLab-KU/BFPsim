docker build . -t $(whoami)/bfpsim:tb
cd ..
docker kill $(whoami)_bfpsim_tb
docker run --rm --user "$(id -u):$(id -g)" -p "$1":8088 -d --name=$(whoami)_bfpsim_tb --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:tb tensorboard --logdir=runs --port 8088 --host 0.0.0.0