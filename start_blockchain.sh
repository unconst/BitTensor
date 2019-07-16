#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source scripts/constant.sh

script="./scripts/init_blockchain.sh"

log "=== stopping blockchain ==="
docker kill eosio_bittensor_container || true

log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name eosio_bittensor_container -d \
-p 8888:8888 -p 9876:9876 \
--mount type=bind,src="$(pwd)"/contract,dst=/opt/eosio/bin/contract \
--mount type=bind,src="$(pwd)"/scripts,dst=/opt/eosio/bin/scripts \
--mount type=bind,src="$(pwd)"/data,dst=/mnt/dev/data \
--mount type=bind,src="$(pwd)"/eos_config,dst=/mnt/dev/config \
-w "/opt/eosio/bin/" $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

if [ "$1" != "--nolog" ]
then
    log "=== follow eosio_bittensor_container logs ==="
    docker logs eosio_bittensor_container --follow
fi
