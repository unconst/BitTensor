#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

source scripts/constant.sh

identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="127.0.0.1"
port=$(jot -r 1  2000 65000)
tensorboard_port=$((port+1))
eosurl="http://127.0.0.1:8888"
logdir='.'

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl $logdir"
log "$COMMAND"


if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
  log "=== building bittensor container ==="
  docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
fi


if [[ "$(docker ps -a | grep bitensor-$identity)" ]]; then
  log "=== stopping bitensor-$identity ==="
  docker stop bitensor-$identity || true
  docker rm bitensor-$identity || true
fi


log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
docker run --rm --name bitensor-$identity -d \
--network="host" \
-p $port:$port \
-p $tensorboard_port:$tensorboard_port \
--mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
--mount type=bind,src="$(pwd)"/src,dst=/bittensor/src \
$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

log "=== follow bittensor_container logs ==="
docker logs bitensor-$identity --follow
