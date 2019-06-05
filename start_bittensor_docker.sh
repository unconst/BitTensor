#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source constant.sh

script="./scripts/init.sh"

identity=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
address="127.0.0.1"
port=$(jot -r 1  2000 65000)
eosurl="http//127.0.0.0:8888"

echo "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name bittensor_container -d \
-e IDENTITY=$identity \
-e ADDRESS=$address \
-e PORT=$port \
-e EOSURL=$eosurl \
--network="host" \
--mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
--mount type=bind,src="$(pwd)"/src,dst=/bittensor/src \
$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

echo "=== follow bittensor_container logs ==="
docker logs bittensor_container --follow
