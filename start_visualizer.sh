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
  echo "Script for starting Visualization instance."
  echo "Usage ./start_visualizer.sh [OPTIONS]"
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
# Bind and advertise this port.
# This port SHOULD REMAIN STATIC as this will be used by ALL nodes 
# to report their findings
port=14142
# Tensorboard port.
tbport=14143

logdir="data/visualizer_container/logs"

# Read command line args
while test 5 -gt 0; do
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
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./visualizer/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep visualizer_container)" ]]; then
    log "=== stopping visualizer_container ==="
    docker stop visualizer_container || true
    docker rm visualizer_container || true
  fi



  # Trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== stop visualizer_container ==="
    docker stop visualizer_container

    exit 0
  }

  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  # Build tensorboard command.
  script="./scripts/visualizer/tensorboard.sh"
  COMMAND="$script $bind_address $port $tbport $logdir"
  log "Run command: $COMMAND"

  log "=== run the docker container locally ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name visualizer_container -d  -t \
  -p $port:$port \
  -p $tbport:$tbport \
  --mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
  --mount type=bind,src="$(pwd)"/data/cache,dst=/bittensor/cache \
  --mount type=bind,src="$(pwd)"/neurons,dst=/bittensor/neurons \
  --mount type=bind,src="$(pwd)"/visualizer,dst=/bittensor/visualizer \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"


  log "=== follow logs ==="
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

  start_local_service
}

main