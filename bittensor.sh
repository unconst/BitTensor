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

function print_help () {
  echo "Script for starting Bittensor instances."
  echo "Usage ./start_bittensor.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help       Print this help message and exit"
  echo " -l, --logdir     Logging directory."
  echo " -p, --port       Server side port for accepting requests."
  echo " -e, --eosurl     URL for EOS blockchain isntance."
  echo " -r, --remote      Run instance locally."
  echo " -t, --token      Digital ocean API token."
}

# [Default Arguments] #
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
# Bind the grpc server to this address with port
bind_address="0.0.0.0"
# Advertise this address on the EOS chain.
serve_address="127.0.0.1"
# Bind and advertise this port.
port=$(( ( RANDOM % 60000 ) + 5000 ))
# Tensorboard port.
tbport=$((port+1))
# URL of eos chain for pulling updates.
eosurl="http://0.0.0.0:8888"
# Directory for sinking logs and model updates.
# TODO(const) Should be root dir.
logdir="data/$identity/logs"
# Is this service running on digital ocean.
remote="false"
# Digital ocean API token for creating remote instances.
token="none"

# Read command line args
while test 7 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -i|--identity)
      identity=`echo $2`
      shift
      shift
      ;;
    -p|--port)
      port=`echo $2`
      tbport=$((port+1))
      shift
      shift
      ;;
    -e|--eosurl)
      eosurl=`echo $2`
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

function init_host() {
  log "init_host"
  # Build droplet if running remotely
  if [ "$remote" == "true" ]; then
    # Create DO instance.
    if [[ "$(docker-machine ls | grep bittensor-$identity)" ]]; then
      log "bittensor-$identity droplet already exists."
    else
      log "Creating Droplet: bittensor-$identity"
      DROPLET_CREATE="docker-machine create --driver digitalocean --digitalocean-size s-4vcpu-8gb --digitalocean-access-token ${token} bittensor-$identity"
      log "Create command: $DROPLET_CREATE"
      eval $DROPLET_CREATE
    fi

    # Set docker context to droplet.
    eval $(docker-machine env bittensor-$identity)
  else
    log "Running bittensor-$identity locally".
  fi
}

function init_image () {
  log "init_image"
  # Init image if non-existent.
  log "=== building bittensor image. ==="
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  else
    # Build anyway.
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG .
  fi
}

function start_service () {

  # Set docker context to droplet.
  if [ "$remote" == "true" ]; then
    eval $(docker-machine env bittensor-$identity)
  fi

  # Stopping instance if already existent.
  if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
    log "=== stopping bittensor-$identity ==="
    docker stop bittensor-$identity || true
    docker rm bittensor-$identity || true
  fi


  # Build start command.
  script="./scripts/bittensor.sh"
  COMMAND="$script $identity $bind_address $port $tbport $eosurl $logdir"
  log "Run command: $COMMAND"


  # Run docker service.
  log "=== run docker container from the $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG image ==="
  if [ "$remote" == "false" ]; then
    docker run --rm --name bittensor-$identity -d  -t \
    --network="host" \
    --mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
    --mount type=bind,src="$(pwd)"/src,dst=/bittensor/src \
    $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
  else
    docker run --rm --name bittensor-$identity -d  -t \
    --network="host" \
    $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
  fi

  docker logs bittensor-$identity --follow


  if [ "$remote" == "true" ]; then
    echo "To tear down this host run:"
    echo "  docker-machine stop bittensor-$identity & docker-machine rm bittensor-$identity --force "
  fi

  # Trap control C (for clean docker container tear down.)
  # function teardown() {
  #   log "=== stop bittensor_container ==="
  #   docker stop bittensor-$identity
  #   eval $(docker-machine env -u)
  #
  #   if [ "$remote" == "true" ]; then
  #     echo "To tear down this host run:"
  #     echo "  docker-machine rm bittensor-$identity --force  "
  #   fi
  #   exit 0
  # }
  # # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  # trap teardown INT SIGHUP SIGINT SIGTERM

}

# Main function.
function main() {

  log "identity: $identity"
  log "remote: $remote"
  log "eosurl: $eosurl"
  log "port: $port"
  log "tbport: $tbport"
  log "logdir: $logdir"

  if [ "$remote" == "true" ]; then
    if [ "$token" == "none" ]; then
      failure "Error: token is none but requesting remote host."
      failure "Visit: https://cloud.digitalocean.com/account/api/tokens"
      exit 0
    fi
  fi

  init_host

  init_image

  start_service

}

main
