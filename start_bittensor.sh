#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"
source ./scripts/constant.sh

# Run under fake random ID and port for localhost testing.
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="0.0.0.0"
port=$(( ( RANDOM % 60000 ) + 5000 ))
tbport=$((port+1))
eosurl="http://0.0.0.0:8888"
logdir="data/$identity/logs"

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $tbport $eosurl $logdir"

log "$COMMAND"
eval $COMMAND
