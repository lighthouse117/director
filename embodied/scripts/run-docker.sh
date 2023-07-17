#!/bin/bash

# Run a docker container

IMAGE_NAME="iiyama/director"
CONTAINER_NAME="iiyama-director1"

# Run the container and install the python requirements

docker run -it --rm --runtime=nvidia --gpus device=1 \
    -v ~/director/embodied:/src/embodied \
    --ipc=host \
    --name $CONTAINER_NAME $IMAGE_NAME \
    /bin/bash

    # -p 6006:6006 \