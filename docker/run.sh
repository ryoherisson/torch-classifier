#!/bin/sh
#
# Run the docker container.

. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -it \
  --gpus all \
  -v $PWD:/workspace \
  --name $CONTAINER_NAME\
  --rm \
  -p 8501:8501 \
  --shm-size=2g \
  $IMAGE_NAME