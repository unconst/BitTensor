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
      echo "starts a single bittensor instance."
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
