#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

# Load constants.
source scripts/constant.sh
source scripts/check_requirements.sh

# Default args.
remote="false"
token="none"

# Commandline args.
while test 3 -gt 0; do
  case "$1" in
    -h|--help)
      echo "Script for starting an Bittensor-EOS chain instance."
      echo "Usage ./start_eos.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo " -h, --help         Print these comments and exit."
      echo " -r, --remote       Run instance on a remote digital ocean instance."
      echo " -t, --token        If -r is set: Use this token to create instance."
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
#

function run_local() {
  # Run the EOS chain through a local docker instance.

  # Build image if not existent.
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "=== building eos container ==="
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  fi

  # Kill already running instance.
  if [[ "$(docker ps -a | grep eos_container)" ]]; then
    log "=== stopping eos ==="
    docker kill eos_container || true
    docker rm eos_container || true
  fi

  # Run the local container with start script.
  log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
  script="./scripts/init_eos.sh"
  docker run --rm --name eos_container -d \
  -p 8888:8888 -p 9876:9876 \
  --mount type=bind,src="$(pwd)"/contract,dst=/opt/eosio/bin/contract \
  --mount type=bind,src="$(pwd)"/scripts,dst=/opt/eosio/bin/scripts \
  --mount type=bind,src="$(pwd)"/data,dst=/mnt/dev/data \
  --mount type=bind,src="$(pwd)"/eos_config,dst=/mnt/dev/config \
  -w "/opt/eosio/bin/" $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

  # Follow logs from the container.
  log "=== follow eos_container logs ==="
  docker logs eos_container --follow
}

function run_remote() {
  # Run the EOS chain on a remote digital ocean instance.

  # Create a digital ocean instance if non-existent.
  # This uses the $TOKEN arg passed as a command line argument.
  if [[ "$(docker-machine ls | grep eosremote)" ]]; then
    log "eos_remote machine already exists"
  else
    log "Creating Droplet:eosremote"
    DM_CREATE="docker-machine create --driver digitalocean --digitalocean-size s-2vcpu-2gb --digitalocean-access-token ${token} eosremote"
#    DM_CREATE="docker-machine create --driver digitalocean --digitalocean-access-token ${token} eosremote"
    log $DM_CREATE
    eval $DM_CREATE
  fi

  # Print instance IP and port for connection.
  EXTERN_IP=$(eval docker-machine ip eosremote)
  log "eosurl: $EXTERN_IP:8888"

  # Set docker machine env to the created host.
  eval $(docker-machine env eosremote)

  # Build the container if non-existent.
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "=== building eos container ==="
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  else
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  fi


  # Stop running container if existent.
  if [[ "$(docker ps -a | grep eos_container)" ]]; then
    log "=== stopping eos ==="
    docker kill eos_container || true
    docker rm eos_container || true
  fi

  # Run start up script through docker on host.
  log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
  script="./scripts/init_eos.sh"
  docker run --rm --name eos_container -d \
  -p 8888:8888 -p 9876:9876 \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$script"

  # Follow eos instance logs.
  log "=== follow eos_container logs ==="
  docker logs eos_container --follow

  # Clean destroy instance.
  docker stop eos_container
  docker rm eos_container
  docker-machine kill eosremote
  docker-machine rm eosremote --force

  # Unset docker-machine environment.
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
