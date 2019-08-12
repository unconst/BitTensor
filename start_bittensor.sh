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
remote="false"
token="none"

# Read command line args
while test 7 -gt 0; do
  case "$1" in
    -h|--help)
      echo "Script for starting Bittensor instances."
      echo "Usage ./start_bittensor.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo " -i, --identity   Node identity."
      echo " -h, --help       Print this help message and exit"
      echo " -r, --remote     Run instance remotely on Digital Ocean."
      echo " -p, --port       Server side port for accepting requests."
      echo " -e, --eosurl     URL for EOS blockchain isntance."
      echo " -l, --logdir     Logging directory."
      echo " -t, --token      Digital ocean API token."
      exit 0
      ;;
    -i|--identity)
      identity=`echo $2`
      shift
      ;;
    -p|--port)
      port=`echo $2`
      tbport=$((port+1))
      shift
      ;;
    -e|--eosurl)
      eosurl=`echo $2`
      shift
      ;;
    -l|--logdir)
      logdir=`echo $2`
      shift
      ;;
    -r|--remote)
      remote="true"
      shift
      ;;
    -t|--token)
      token=`echo $2`
      shift
      ;;
    *)
      break
      ;;
  esac
done

script="./scripts/bittensor.sh"
COMMAND="$script $identity $address $port $tbport $eosurl $logdir"

log "$COMMAND"
eval $COMMAND
