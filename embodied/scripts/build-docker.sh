#!/bin/bash

# Build docker image

IMAGE_NAME="iiyama/director"

docker build -t $IMAGE_NAME .
