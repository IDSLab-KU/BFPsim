mkdir saves
mkdir stats
mkdir logs

docker build . -t $(whoami)/floatblock:latest
