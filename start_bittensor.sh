#!/usr/bin/env bash
set -o errexit

source constant.sh

# change to script's directory
cd "$(dirname "$0")"

identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="localhost"
port=$(jot -r 1  2000 65000)
eosurl="http://localhost:8888"

source constant.sh
script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl"

eval $COMMAND
