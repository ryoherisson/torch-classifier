#!/bin/sh
#
# Build the tensorboard.

docker build -f ./docker/tensorboard.Dockerfile -t tensorboard .