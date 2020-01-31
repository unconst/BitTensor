#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

# Load constants
source scripts/constant.sh

#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"
source ./scripts/constant.sh

# Check script check_requirements
source scripts/check_requirements.sh

function print_help () {
  echo "Script for starting Tensorboard instance."
  echo "Usage ./tensorboard.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help       Print this help message and exit"
  echo " -l, --logdir     Logging directory."
  echo " -p, --port       Bind side port for accepting requests."
  echo " -r, --remote     Run instance locally."
  echo " -t, --token      Digital ocean API token."
}

# [Default Arguments] #
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
# Bind the grpc server to this address with port
bind_address="0.0.0.0"

# Advertise this address on the EOS chain.
machine=$(whichmachine)
echo "Detected host: $machine"
if [[ "$machine" == "Darwin" ||  "$machine" == "Mac" ]]; then
    serve_address="host.docker.internal"
else
    serve_address="172.17.0.1"
fi


# Read command line args
while test 9 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -p|--port)
      port=`echo $2`
      tbport=$((port+1))
      shift
      shift
      ;;
    -l|--logdir)
      logdir=`echo $2`
      shift
      shift
      ;;
    -r|--remote)
      remote="true"
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

function start_local_service() {
  log "=== run locally. ==="

  # Init image if non-existent.
  log "=== building bittensor image. ==="

  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./tensorboard/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./tensorboard/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep tensorboard_container)" ]]; then
    log "=== stopping tensorboard_container ==="
    docker stop tensorboard_container || true
    docker rm tensorboard_container || true
  fi

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== stop bittensor_container ==="
    docker stop tensorboard_container

    exit 0
  }

  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  # Build start command.
  script="./scripts/tensorboard.sh"
}

function main() {
  log "%%%%%%%%.%%%%%%%.%.....%..%%%%%..%%%%%%%.%%%%%%..%%%%%%..%%%%%%%....%....%%%%%%..%%%%%%.."
  log "...%....%.......%%....%.%.....%.%.....%.%.....%.%.....%.%.....%...%.%...%.....%.%.....%."
  log "...%....%.......%.%...%.%.......%.....%.%.....%.%.....%.%.....%..%...%..%.....%.%.....%."
  log "...%....%%%%%...%..%..%..%%%%%..%.....%.%%%%%%..%%%%%%..%.....%.%.....%.%%%%%%..%.....%."
  log "...%....%.......%...%.%.......%.%.....%.%...%...%.....%.%.....%.%%%%%%%.%...%...%.....%."
  log "...%....%.......%....%%.%.....%.%.....%.%....%..%.....%.%.....%.%.....%.%....%..%.....%."
  log "...%....%%%%%%%.%.....%..%%%%%..%%%%%%%.%.....%.%%%%%%..%%%%%%%.%.....%.%.....%.%%%%%%.."

  start_local_service
}

main