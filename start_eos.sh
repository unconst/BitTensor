#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

# Load constants.
source scripts/constant.sh

# Read command line args
remote="false"
token="none"
while test 3 -gt 0; do
  case "$1" in
    -h|--help)
      echo "starts a single eos instance"
      exit 0
      ;;
    -r|--remote)
      remote="true"
      shift
      ;;
    -t|--token)
      token=$2
      shift
      ;;
    *)
      break
      ;;
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
    docker rm eos_container || true
  fi


  log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
  docker run --rm --name eos_container -d \
  -p 8888:8888 -p 9876:9876 \
  --mount type=bind,src="$(pwd)"/contract,dst=/opt/eosio/bin/contract \
  --mount type=bind,src="$(pwd)"/scripts,dst=/opt/eosio/bin/scripts \
  --mount type=bind,src="$(pwd)"/data,dst=/mnt/dev/data \
  --mount type=bind,src="$(pwd)"/eos_config,dst=/mnt/dev/config \
  -w "/opt/eosio/bin/" $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

  log "=== follow eos_container logs ==="
  docker logs eos_container --follow
}

function run_remote() {

  if [[ "$(docker-machine ls | grep eosremote)" ]]; then
    log "eos_remote machine already exists"
  else
    log "Creating Droplet:eos_remote"
    DM_CREATE="docker-machine create --driver digitalocean --digitalocean-access-token ${token} eosremote"
    log $DM_CREATE
    eval $DM_CREATE
  fi

  EXTERN_IP=$(eval docker-machine ip eosremote)
  log "eosurl: $EXTERN_IP:8888"

  eval $(docker-machine env eosremote)

  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "=== building eos container ==="
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  fi

  if [[ "$(docker ps -a | grep eos_container)" ]]; then
    log "=== stopping eos ==="
    docker kill eos_container || true
    docker rm eos_container || true
  fi

  script="./scripts/init_eos.sh"
  docker run --rm --name eos_container -d \
  -p 8888:8888 -p 9876:9876 \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

  log "=== follow eos_container logs ==="
  docker logs eos_container --follow

  eval "$(docker-machine env -u)"
}


function main() {

  if [ "$remote" == "false" ]; then
      log "Running eos Local."
      run_local
  else
      log "Running eos Remote."
      run_remote
  fi
}

main
