#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"
source ./scripts/constant.sh

# Kill subprocesses.
function clean_up {

    # Perform program exit housekeeping
    KILL $tensorboard_pid
    exit
}
trap clean_up SIGHUP SIGINT SIGTERM


# Run under fake random ID and port for localhost testing.
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="localhost"
port=$(( ( RANDOM % 60000 ) + 5000 ))
tensorboard_port=$(port + 1)
eosurl="http://localhost:8888"
logdir="data/$identity/logs"

log "tensorboard --logdir=$logdir --port=$tensorboard_port"
log "Tensorboard: http://0.0.0.0:$tensorboard_port"
tensorboard --logdir=$logdir --port=$tensorboard_port &
tensorboard_pid=$!

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl $logdir"

log "$COMMAND"
eval $COMMAND
