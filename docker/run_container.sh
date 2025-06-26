docker rm -f foundationpose
DIR=$(pwd)/../
xhost local:1000 + && \
docker run \
    --name foundationpose  \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt  \
    -v /tmp:/tmp  \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --network=host \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE foundationpose:latest bash -c "cd $DIR && bash"