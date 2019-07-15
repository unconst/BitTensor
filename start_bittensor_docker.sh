#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source scripts/constant.sh

identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="127.0.0.1"
port=$(jot -r 1  2000 65000)
eosurl="http://127.0.0.1:8888"
logdir=''

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl $logdir"
log "$COMMAND"

log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name bitensor-$identity -d \
--network="host" \
--mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
--mount type=bind,src="$(pwd)"/src,dst=/bittensor/src \
$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

log "=== follow bittensor_container logs ==="
docker logs bitensor-$identity --follow
