#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

# Load constants.
source scripts/constant.sh

verbose='true'
remote='false'
while getopts 'rv' flag; do
  case "${flag}" in
    r) remote='true' ;;
    v) verbose='true' ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

function run_local() {
  script="./scripts/init_eos.sh"
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "=== building eos container ==="
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  fi


  if [[ "$(docker ps -a | grep eos_container)" ]]; then
    log "=== stopping eos ==="
    docker kill eos_container || true
  fi


  log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
  docker run --rm --name eos_container -d \
  -p 8888:8888 -p 9876:9876 \
  --mount type=bind,src="$(pwd)"/contract,dst=/opt/eosio/bin/contract \
  --mount type=bind,src="$(pwd)"/scripts,dst=/opt/eosio/bin/scripts \
  --mount type=bind,src="$(pwd)"/data,dst=/mnt/dev/data \
  --mount type=bind,src="$(pwd)"/eos_config,dst=/mnt/dev/config \
  -w "/opt/eosio/bin/" $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

  if [ "$verbose" == "true" ]
  then
      log "=== follow eos_container logs ==="
      docker logs eos_container --follow
  fi
}


function run_remote() {
  # TODO(const) implement remote running.
  a=1
}


function main() {

  if [ "$remote" == "false" ]; then
      log "running eos local."
      run_local
  else
      log "running eos remote."
      run_remote
  fi
}

main
