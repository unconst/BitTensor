#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"
source ./scripts/constant.sh

# Check script check_requirements
source scripts/check_requirements.sh

function print_help () {
  echo "Script for starting Visualization instance."
  echo "Usage ./start_visualizer.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help            Print this help message and exit"
  echo " -c, --config_path     Path to config yaml."
  echo " -e, --eos_url         Path to EOS chain."
  echo " -r, --remote     Run instance locally."
  echo " -t, --token      Digital ocean API token."
}

config_path='visualizer/config.yaml'

# bind and advertise this port
port=$(( ( RANDOM % 60000 ) + 5000 ))

# Is this service running on digital ocean.
remote="false"
# Digital ocean API token for creating remote instances.
token="none"

# Read command line args
while test 5 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -c|--config_path)
      config_path=`echo $2`
      shift
      shift
      ;;
    -e|--eos_url)
      eos_url=`echo $2`
      shift
      shift
      ;;
    -r|--remote)
      remote="true"
      shift
      shift
      ;;
    -t|--token)
      token=`echo $2`
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

# read yaml file
eval $(parse_yaml "$config_path" "config_")

function start_remote_service() {
  log "=== run remote. ==="

  # Build trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== tear down. ==="
    eval $(docker-machine env -u)
    echo "To tear down this host run:"
    echo "  docker-machine stop visualizer-container & docker-machine rm visualizer-container --force "
    exit 0
  }
  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  # Initialize the host.
  log "=== initializing remote host. ==="
  if [[ "$(docker-machine ls | grep visualizer-container)" ]]; then
    # Host already exists.
    log "visualizer-container droplet already exists."
  else
    log "Creating Droplet: visualizer-container"
    DROPLET_CREATE="docker-machine create --driver digitalocean --digitalocean-size s-4vcpu-8gb --digitalocean-access-token ${token} visualizer-container"
    log "Create command: $DROPLET_CREATE"
    eval $DROPLET_CREATE
  fi

  # Set the docker context to the droplet.
  log "=== switching droplet context. ==="
  eval $(docker-machine env visualizer-container)

  # Build the image.
  # Init image if non-existent.
  log "=== building visualizer image. ==="
  #docker pull unconst/bittensor:latest
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep visualizer-container)" ]]; then
    log "=== stopping visualizer-container ==="
    docker stop visualizer-container || true
    docker rm visualizer-container|| true
  fi

  # Find the external ip address for this droplet.
  serve_address=$(eval docker-machine ip visualizer-container)
  log "serve_address: $serve_address:$port"

  # Build start command.
  script=$config_script$
  COMMAND="$config_script --config_path $config_path --eos_url $eos_url"
  log "Command: $COMMAND"
  log "Image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"

  # Run docker service.
  log "=== run the docker container on remote host. ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name visualizer-container -d  -t \
  -p $port:$port \
  -p $config_tbport:$config_tbport \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

  log "=== follow ==="
  docker logs visualizer-container --follow
}

function start_local_service() {
  log "=== run locally. ==="
  log ""

  # Init image if non-existent.
  log "=== build image. ==="
  log ""

  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep visualizer-container)" ]]; then
    log "=== stop visualizer-container ==="
    docker stop visualizer-container || true
    docker rm visualizer-container || true
    log ""
  fi

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== stop visualizer-container ==="
    docker stop visualizer-container
    log ""

    exit 0
  }

  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  echo "Monitoring chain at $eos_url"

  # Build start command.
  log "=== run container === "

  script=$config_script$
  COMMAND="$config_script --config_path $config_path --eos_url $eos_url"
  log "Command: $COMMAND"
  log "Image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"

  docker run --rm --name visualizer-container -d  -t \
  -p $config_port:$config_port \
  -p $config_tbport:$config_tbport \
  --mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
  --mount type=bind,src="$(pwd)"/data/visualizer/logs,dst=/bittensor/data/visualizer/logs \
  --mount type=bind,src="$(pwd)"/neurons,dst=/bittensor/neurons \
  --mount type=bind,src="$(pwd)"/visualizer,dst=/bittensor/visualizer \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
  log ""

  docker logs visualizer-container --follow
}

function main() {
  log "%%%%%%%%.%%%%%%.%.....%..%%%%%..%%%%%%%.%%%%%%..%%%%%%..%%%%%%%....%....%%%%%%..%%%%%%.."
  log "...%....%.......%%....%.%.....%.%.....%.%.....%.%.....%.%.....%...%.%...%.....%.%.....%."
  log "...%....%.......%.%...%.%.......%.....%.%.....%.%.....%.%.....%..%...%..%.....%.%.....%."
  log "...%....%%%%%...%..%..%..%%%%%..%.....%.%%%%%%..%%%%%%..%.....%.%.....%.%%%%%%..%.....%."
  log "...%....%.......%...%.%.......%.%.....%.%...%...%.....%.%.....%.%%%%%%%.%...%...%.....%."
  log "...%....%.......%....%%.%.....%.%.....%.%....%..%.....%.%.....%.%.....%.%....%..%.....%."
  log "...%....%%%%%%%.%.....%..%%%%%..%%%%%%%.%.....%.%%%%%%..%%%%%%%.%.....%.%.....%.%%%%%%.."
  log "........................................................................................"


  log "remote: $remote"
  log "eosurl: $eos_url"
  log "logdir: $logdir"
  log "token: $token"

  if [ "$remote" == "true" ]; then
    if [ "$token" == "none" ]; then
      failure "Error: token is none but requesting remote host."
      failure "Visit: https://cloud.digitalocean.com/account/api/tokens"
      exit 0
    fi
  fi

  if [ "$remote" == "true" ]; then
    if [ "$upnpc" == "true" ]; then
      failure "Error: cannot port map on remote hosts"
      exit 0
    fi
  fi

  if [ "$remote" == "true" ]; then
    start_remote_service
  else
    start_local_service
  fi
}

main
