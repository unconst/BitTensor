#!/usr/bin/env bash
source ./scripts/constant.sh

function print_help () {
  echo "Script for starting Visualization instance."
  echo "Usage ./visualizer.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help            Print this help message and exit"
  echo " -c, --config_path     Path to config yaml."
  echo " -e, --eos_url         EOS chain path"     
}

config_path='visualizer/config.yaml'
eos_url='docker.host.internal'

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


function start_tensorboard() {
  log "=== start Tensorboard ==="
  log "Command: tensorboard --logdir=$config_logdir --port=$config_tbport --host=$config_bind_address"
  log "Endpoint: http://$config_bind_address:$config_tbport"
  log ""
  tensorboard --logdir=$config_logdir --port=$config_tbport --host=$config_bind_address &
  TensorboardPID=$!
}

function start_node_listener() {

  log "=== start Visualizer ==="
  COMMAND="python3 visualizer/main.py --config_path=$config_path --eos_url=$eos_url"
  log "Command: $COMMAND"
  eval $COMMAND &
  LISTENERPID=$!
  log ""

}

# Clean docker tear down.
function teardown() {
  # Perform program exit & housekeeping
  kill -9 $TensorboardPID
  log "=== stopped Tensorboard ==="

  kill -9 $LISTENERPID
  log "=== stopped node listener ==="

  exit 0
}

function main() {

  # Build protos
  ./scripts/build_protos.sh

  # Start Tensorboard.
  start_tensorboard

  # start listening to incoming data from running nodes
  start_node_listener

  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM

  # idle waiting for abort from user
  read -r -d '' _ </dev/tty
}

# Run.
main
