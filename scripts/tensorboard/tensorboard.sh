#!/usr/bin/env bash
source ./scripts/constant.sh

# Arguments to this script.

# Local address to bind our server to.
BIND_ADDRESS=$1
# Port to bind endpoint on.
PORT=$2
# Port to bind Tensorboard on.
TBPORT=$3
# Directory to save checkpoints and logs.
LOGDIR=$4

function start_tensorboard() {
  log "=== start Tensorboard ==="
  log "tensorboard --logdir=$LOGDIR --port=$TBPORT --host=$BIND_ADDRESS"
  log "Tensorboard: http://$BIND_ADDRESS:$TBPORT"
  tensorboard --logdir=$LOGDIR --port=$TBPORT --host=$BIND_ADDRESS &
  TensorboardPID=$!
}

function start_node_listener() {

  COMMAND="python3 ./scripts/tensorboard/node_stats_listener.py $PORT $LOGDIR"
  log "Running command: $COMMAND"
  eval $COMMAND &
  LISTENERPID=$!
}

function main() {

  # Intro logs.
  log "=== Tensorboard ==="
  log "Args {"
  log "   BIND_ADDRESS: $BIND_ADDRESS"
  log "   PORT: $PORT"
  log "   TBPORT: $TBPORT"
  log "   LOGDIR: $LOGDIR"
  log "}"
  log ""
  log "=== setup accounts ==="

  # Build protos
  ./scripts/build_protos.sh

  # Start Tensorboard.
  start_tensorboard

  # start listening to incoming data from running nodes
  start_node_listener

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    # Perform program exit & housekeeping
    kill -9 $TensorboardPID
    log "=== stopped Tensorboard ==="

    kill -9 $LISTENERPID
    log "=== stopped node listener ==="


    exit 0
  }
  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM

  # idle waiting for abort from user
  read -r -d '' _ </dev/tty


}

# Run.
main


