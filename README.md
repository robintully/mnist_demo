Loading bash shell in docker image -
    docker run -it mnist_web   /bin/bash


Copying a file from docker -
    docker cp <containerId>:/file/path/within/container /host/path/target

#TODO: fix docker image for training

#TODO: change file paths to env variables

#TODO: change model storage and import to s3