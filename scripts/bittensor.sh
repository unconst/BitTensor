#!/usr/bin/env bash
source ./scripts/constant.sh

# Arguments to this script.

# Network identity
IDENTITY=$1
# Address to post to the EOS chain. (our API endpoint)
SERVE_ADDRESS=$2
# Local address to bind our server to.
BIND_ADDRESS=$3
# Port to bind endpoint on.
PORT=$4
# Port to bind Tensorboard on.
TBPORT=$5
# URL of EOS endpoint.
EOSURL=$6
# Directory to save checkpoints and logs.
LOGDIR=$7
# Python client to run.
CLIENT=$8

# Creates the system eoisio wallet. This is used to build our unique account.
# In the future the eosio account will be replaced with your own.
function create_eosio(){
  trace "cleos -u $EOSURL wallet create -n eosio --to-console"
  cleos -u $EOSURL wallet create -n eosio --to-console >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "created wallet: eosio."
  else
      failure 'failed to create eosio wallet.'
      cat data/$IDENTITY/bittensor_logs.out 2>&1
  fi
}

# Imports the eosio private key into the eosio wallet.
function import_eosio() {
  trace "cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY"
  cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "imported eosio key."
  else
      failure 'failed to import eosio key.'
      cat data/$IDENTITY/bittensor_logs.out 2>&1
      exit 1
  fi
}

# Unlocks the eosio wallet using the eosio wallet password.
# In the future this will us your wallet own password.
function unlock_eosio() {
  trace "cleos -u $EOSURL wallet unlock -n eosio --password $EOSIO_PASSWORD"
  cleos -u $EOSURL wallet unlock -n eosio --password $EOSIO_PASSWORD >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "unlocked eosio."
  else
      success 'unlocked eosio.'
  fi
}

# Creates an account on the eos blockchain and assigns the eosio pub key as
# owner and active key giving us permission to tranfer it's funds and make
# contract transactions at a later time.
function create_account() {
  trace "cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY"
  cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "created account: $IDENTITY."
  else
      failure "failed to created account: $IDENTITY. With error above."
      cat data/$IDENTITY/bittensor_logs.out 2>&1
      #exit 1
  fi
}

# Publish our newly formed account into the bittensoracc metagraph. We publish
# our id, address, and port allowing other nodes to communicate with us.
function subscribe_account() {
  trace "cleos -u $EOSURL push action bittensoracc subscribe "["$IDENTITY", "$SERVE_ADDRESS", "$PORT"]" -p $IDENTITY@active"
  cleos -u $EOSURL push action bittensoracc subscribe "["$IDENTITY", "$SERVE_ADDRESS", "$PORT"]" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "subscribe account: $IDENTITY."
  else
      failure "failed to subscribe account: $IDENTITY. Check your EOSURL connection."
      cat data/$IDENTITY/bittensor_logs.out 2>&1
      exit 1
  fi
}

# Unpublish our account in the bittensoracc contract. This signals our leaving
# the network also, it uncluters the network.
function unsubscribe_account() {
  trace "cleos -u $EOSURL push action bittensoracc unsubscribe "["$IDENTITY"]" -p $IDENTITY@active"
  cleos -u $EOSURL push action bittensoracc unsubscribe "["$IDENTITY"]" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "unsubscribe account: $IDENTITY."
  else
      failure "failed to unsubscribe account: $IDENTITY. Check your EOSURL connection."
      cat data/$IDENTITY/bittensor_logs.out 2>&1
      exit 1
  fi
}

# Prints the metagraph state to terminal.
function print_metagraph() {
  trace "cleos get table bittensoracc bittensoracc metagraph"
  log "Metagraph:"
  cleos -u $EOSURL get table bittensoracc bittensoracc metagraph
}


function start_tensorboard() {
  log "=== start Tensorboard ==="
  log "tensorboard --logdir=$LOGDIR --port=$TBPORT --host=$BIND_ADDRESS"
  log "Tensorboard: http://$BIND_ADDRESS:$TBPORT"
  tensorboard --logdir=$LOGDIR --port=$TBPORT --host=$BIND_ADDRESS &
  TensorboardPID=$!
}

function start_neuron() {
  # The main command.
  # Start our Neuron object training, server graph, open dendrite etc.
  log ""
  log "=== start Neuron ==="
  log "python src/main.py $IDENTITY $SERVE_ADDRESS $BIND_ADDRESS $PORT $EOSURL $LOGDIR"
  python "clients/$CLIENT/main.py" $IDENTITY $SERVE_ADDRESS $BIND_ADDRESS $PORT $EOSURL $LOGDIR &
  NueronPID=$!
}

function main() {
  # Create the state directory for logs and model checkpoints.
  # TODO(const) In the future this could be preset and contain our conf file.
  mkdir -p data/$IDENTITY
  touch data/$IDENTITY/bittensor_logs.out

  # Intro logs.
  log "=== BitTensor ==="
  log "Args {"
  log "   EOSURL: $EOSURL"
  log "   IDENTITY: $IDENTITY"
  log "   SERVE_ADDRESS: $SERVE_ADDRESS"
  log "   BIND_ADDRESS: $BIND_ADDRESS"
  log "   PORT: $PORT"
  log "   TBPORT: $TBPORT"
  log "   LOGDIR: $LOGDIR"
  log "   CLIENT: clients/$CLIENT/main.py"
  log "}"
  log ""
  log "=== setup accounts ==="


  # TODO(const) These are currently hard coded to eosio main. In prodution this
  # should absolutely change.
  # Check to see if eosio wallet exists.
  # If not, create eosio account and pull private keys to this wallet.
  echo "EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3"
  echo "EOSIO_PUBLIC_KEY=EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV"
  echo "EOSIO_PASSWORD=PW5JgJBjC1QXf8XoYPFY56qF5SJLLJNfjHbCai3DyC6We1FeBRL8q"
  EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
  EOSIO_PUBLIC_KEY=EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV
  EOSIO_PASSWORD=PW5JgJBjC1QXf8XoYPFY56qF5SJLLJNfjHbCai3DyC6We1FeBRL8q


  # Unlock eosio wallet. Silent failure on 'already unlocked error'.
  # or silent failure on does not exist.
  unlock_eosio

  # Pull the eosio pub key.
  PUBLIC_KEY=$(cleos -u $EOSURL wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')

  # Create eosio wallet if it does not exist.
  if [[ $EOSIO_PUBLIC_KEY != $PUBLIC_KEY ]]; then
    create_eosio
    import_eosio
  fi

  # Create out Identity account on the EOS blockchain. Set ownership to eosio.
  create_account

  # Publish our newly formed account to the eos blockchain.
  subscribe_account

  # Print metagraph.
  print_metagraph

  # Build protos
  ./scripts/build_protos.sh

  # Start Tensorboard.
  start_tensorboard

  # Start eh Neuron object.
  start_neuron

  # Trap control C (for clean docker container tear down.)
  function teardown() {
    # Perform program exit & housekeeping
    kill -9 $TensorboardPID
    log "=== stopped Tensorboard ==="

    kill -9 $NueronPID
    log "=== stopped Nucleus ==="

    unsubscribe_account
    log "=== unsubscribed from Metagraph. ==="

    exit 0
  }
  # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
  trap teardown INT SIGHUP SIGINT SIGTERM

  # idle waiting for abort from user
  read -r -d '' _ </dev/tty
}

# Run.
main
