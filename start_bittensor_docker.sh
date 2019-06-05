#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source constant.sh

script="./scripts/init.sh"

echo "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name bittensor_container -d \
--env-file config.txt \
 --network="host" \
--mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
--mount type=bind,src="$(pwd)"/src,dst=/bittensor/src \
$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

echo "=== follow bittensor_container logs ==="
docker logs bittensor_container --follow
