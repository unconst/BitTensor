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
  echo "Usage ./bittensor.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help       Print this help message and exit"
  echo " -i, --identity   EOS identity."
  echo " -n, --neuron     bittensor neuron name e.g. boltzmann"
  echo " -l, --logdir     Logging directory."
  echo " -p, --port       Bind side port for accepting requests."
  echo " -e, --eosurl     URL for EOS blockchain isntance."
  echo " -r, --remote     Run instance locally."
  echo " -t, --token      Digital ocean API token."
  echo " -m  --upnpc      Port map for NAT."
}

# [Default Arguments] #
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
# Bind the grpc server to this address with port
bind_address="0.0.0.0"
# Advertise this address on the EOS chain.
serve_address="host.docker.internal"
# Bind and advertise this port.
port=$(( ( RANDOM % 60000 ) + 5000 ))
# Tensorboard port.
tbport=$((port+1))
# URL of eos chain for pulling updates. DEFAULT to localhost on host.
eosurl="http://host.docker.internal:8888"
# Directory for sinking logs and model updates.
# TODO(const) Should be root dir.
logdir="data/$identity/logs"
# Is this service running on digital ocean.
remote="false"
# Digital ocean API token for creating remote instances.
token="none"
# Neuron: The client adhering to the Bittensor protocol.
neuron="Mach"
# Upnpa: should map ports on Nat. This creates a whole in your router.
upnpc="false"

# Read command line args
while test 9 -gt 0; do
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
    -n|--neuron)
      neuron=`echo $2`
      shift
      shift
      ;;
    -m|--upnpc)
      upnpc="true"
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
  #docker pull unconst/bittensor:latest
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./neurons/$neuron/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./neurons/$neuron/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
    log "=== stopping bittensor-$identity ==="
    docker stop bittensor-$identity || true
    docker rm bittensor-$identity || true
  fi

  # Create port mappings
  if [ "$upnpc" == "true" ]; then
    log "=== creating TCP tunnel in your router. ==="
    log "Installing miniupnpc"
    pip install miniupnpc
    log "Punching hole in router."
    external_ip_address_and_port=$(python scripts/upnpc.py --port $port)
    return_code="$(cut -d':' -f1 <<< "$external_ip_address_and_port")"
    external_ip="$(cut -d':' -f2 <<< "$external_ip_address_and_port")"
    external_port="$(cut -d':' -f3 <<< "$external_ip_address_and_port")"
    if [ "$return_code" == "success" ]; then
      log "setting server_address to $external_ip"
      serve_address=$external_ip
      log "setting port to $external_port"
      port=$external_port
    else
      log "failure during port mapping."
      log "Does your router support UPNPC?"
      log "You may need to manually punch a hole in your router."
      log "=== deleting TCP tunnel in your router. ==="
      python scripts/upnpc.py --port $port --delete True
      exit
    fi
  fi

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== stop bittensor_container ==="
    docker stop bittensor-$identity

    # deleting the port mapping.
    if [ "$upnpc" == "true" ]; then
      log "=== deleting TCP tunnel in your router. ==="
      python scripts/upnpc.py --port $port --delete True
    fi

    exit 0
  }
  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  # Build start command.
  script="./scripts/bittensor.sh"
  COMMAND="$script $identity $serve_address $bind_address $port $tbport $eosurl $logdir $neuron"
  log "Run command: $COMMAND"

  # Run docker service.
  log "=== run the docker container locally ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name bittensor-$identity -d  -t \
  -p $port:$port \
  -p $tbport:$tbport \
  --mount type=bind,src="$(pwd)"/scripts,dst=/bittensor/scripts \
  --mount type=bind,src="$(pwd)"/data/cache,dst=/bittensor/cache \
  --mount type=bind,src="$(pwd)"/neurons,dst=/bittensor/neurons \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

  log "=== follow logs ==="
  docker logs bittensor-$identity --follow
}

function start_remote_service() {
  log "=== run remote. ==="

  # Build trap control C (for clean docker container tear down.)
  function teardown() {
    log "=== tear down. ==="
    eval $(docker-machine env -u)
    echo "To tear down this host run:"
    echo "  docker-machine stop bittensor-$identity & docker-machine rm bittensor-$identity --force "
    exit 0
  }
  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

  # Initialize the host.
  log "=== initializing remote host. ==="
  if [[ "$(docker-machine ls | grep bittensor-$identity)" ]]; then
    # Host already exists.
    log "bittensor-$identity droplet already exists."
  else
    log "Creating Droplet: bittensor-$identity"
    DROPLET_CREATE="docker-machine create --driver digitalocean --digitalocean-size s-4vcpu-8gb --digitalocean-access-token ${token} bittensor-$identity"
    log "Create command: $DROPLET_CREATE"
    eval $DROPLET_CREATE
  fi

  # Set the docker context to the droplet.
  log "=== switching droplet context. ==="
  eval $(docker-machine env bittensor-$identity)

  # Build the image.
  # Init image if non-existent.
  log "=== building bittensor image. ==="
  #docker pull unconst/bittensor:latest
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./neurons/$neuron/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./neurons/$neuron/Dockerfile .
  fi

  # Stop the container if it is already running.
  if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
    log "=== stopping bittensor-$identity ==="
    docker stop bittensor-$identity || true
    docker rm bittensor-$identity || true
  fi

  # Find the external ip address for this droplet.
  serve_address=$(eval docker-machine ip bittensor-$identity)
  log "serve_address: $serve_address:$port"

  # Build start command.
  script="./scripts/bittensor.sh"
  COMMAND="$script $identity $serve_address $bind_address $port $tbport $eosurl $logdir $neuron"
  log "Run command: $COMMAND"

  # Run docker service.
  log "=== run the docker container on remote host. ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name bittensor-$identity -d  -t \
  -p $port:$port \
  -p $tbport:$tbport \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

  log "=== follow ==="
  docker logs bittensor-$identity --follow
}

# Main function.
function main() {

  log "identity: $identity"
  log "remote: $remote"
  log "eosurl: $eosurl"
  log "port: $port"
  log "server_address: $serve_address"
  log "bind_address: $bind_address"
  log "tbport: $tbport"
  log "logdir: $logdir"
  log "neuron: $neuron"
  log "upnpc: $upnpc"

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
