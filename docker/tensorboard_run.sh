#!/bin/sh
#
# Build the tensorboard.

docker run --rm -it -v /logs:/logs -p 6006:6006 tensorboard