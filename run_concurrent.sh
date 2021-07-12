# ecution
## Set config name here
conf_names=(MobileNetv1_CIFAR100_FB12 ResNet18_CIFAR100_FB12 VGG16_CIFAR100_FB12 AlexNet_CIFAR100_FB12)
## Please Change Session name every time, it will override existing session.
session=fb0

# docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app floatblock:latest python3 -u /app/main.py -m VGG16 -bf VGG16_8

# ================ Do not touch below here ==================
cpus="8"
user="$(id -u):$(id -g)"
devices=('"device=0"' '"device=1"' '"device=2"' '"device=3"' '"device=4"' '"device=5"' '"device=6"' '"device=7"')

# Create Tmux Session
tmux kill-session -t $session
tmux new -d -s $session
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v

# Send keys to sessions
for i in ${!conf_names[@]}
do
    tmux send-keys -t $session.$i "docker run --rm --gpus '${devices[$i]}' --cpus=${cpus} --user ${user} --workdir /app -v $(pwd):/app lsh/flexblock:latest python3 -u /app/main.py --mode train -tc ${conf_names[$i]} --log True --stat True --save True" C-m
done