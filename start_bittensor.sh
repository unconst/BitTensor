#!/usr/bin/env bash
set -o errexit

# change to script's directory
cd "$(dirname "$0")"

identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
address="127.0.0.1"
port=$(jot -r 1  2000 65000)
eosurl="http://127.0.0.1:8888"

source constant.sh
script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $eosurl"

echo -e $COMMAND
eval $COMMAND
