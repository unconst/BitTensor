#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"
source ./scripts/constant.sh

# Run under fake random ID and port for localhost testing.
identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="localhost"
port=$(jot -r 1  2000 65000)
eosurl="http://localhost:8888"

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl"
log "$COMMAND"
eval $COMMAND
