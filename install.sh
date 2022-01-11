# Installing main
docker build . -t $(whoami)/bfpsim:main

# Installing tensorboard
cd tensorboard
docker build . -t $(whoami)/bfpsim:tensorboard
cd ..
