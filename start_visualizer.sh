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
}

config_path='visualizer/config.yaml'

# Read command line args
while test 4 -gt 0; do
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
      ;;
    *)
      break
      ;;
  esac
done

# read yaml file
eval $(parse_yaml "$config_path" "config_")

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
  if [[ "$(docker ps -a | grep visualizer_container)" ]]; then
    log "=== stop visualizer_container ==="
    docker stop visualizer_container || true
    docker rm visualizer_container || true
    log ""
  fi

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== stop visualizer_container ==="
    docker stop visualizer_container
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

  docker run --rm --name visualizer_container -d  -t \
  -p $config_port:$config_port \
  -p $config_tbport:$config_tbport \
  --mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
  --mount type=bind,src="$(pwd)"/data/visualizer/logs,dst=/bittensor/data/visualizer/logs \
  --mount type=bind,src="$(pwd)"/neurons,dst=/bittensor/neurons \
  --mount type=bind,src="$(pwd)"/visualizer,dst=/bittensor/visualizer \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
  log ""

  docker logs visualizer_container --follow
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

  start_local_service
}

main
