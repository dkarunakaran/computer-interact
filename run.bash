#!/bin/bash
# Run below step one by one in terminal
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
docker build -t broswer_bot .

# Get the current username
username=$(whoami)
if [ "$username" == "beastan" ]; then
    echo "We are running the docker container on $username's computer."
    docker run --net host --gpus all -it -v /media/$username/projects/broswer-bot:/app broswer_bot
elif [ "$username" == "dhanoop" ]; then
    echo "We are running the docker container on $username's computer."
    docker run --net host --gpus all -it -v /home/$username/Documents/projects/broswer-bot:/app broswer_bot
else
    echo "Wrong system to run this script."
fi

