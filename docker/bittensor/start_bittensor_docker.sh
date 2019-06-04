#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source constant.sh

script="./scripts/init_account.sh"

echo "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name bittensor_container -d \
-e IDENTITY=$1 \
-e ADDRESS=$2 \
 --network="host" \
--mount type=bind,src="$(pwd)"/scripts,dst=/opt/bittensor/bin/scripts \
-w "/opt/bittensor/bin/" $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

echo "=== follow bittensor_container logs ==="
docker logs bittensor_container --follow
