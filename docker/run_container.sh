DIR=$(pwd)/
xhost +local:1000

if docker ps -a --format '{{.Names}}' | grep -q '^foundationpose$'; then
    echo "Attaching to existing foundationpose container..."
    docker start foundationpose 2>/dev/null
    docker exec -it -e DISPLAY="$DISPLAY" -w "$(pwd)" foundationpose bash
else
    echo "Starting new foundationpose container..."
    docker run \
        --name foundationpose \
        --gpus all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -it \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "$(pwd):$(pwd)" \
        -v /home:/home \
        -v /mnt:/mnt \
        -v /tmp:/tmp \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
        --network=host \
        --ipc=host \
        -e DISPLAY="$DISPLAY" \
        -w "$(pwd)" \
        foundationpose:latest bash -c "cd $DIR && bash"
fi