#!/usr/bin/env bash

source constant.sh

# Arguments to this script.
IDENTITY=$1
ADDRESS=$2
PORT=$3
EOSURL=$4

function create_eosio(){
  cleos -u $EOSURL wallet create -n eosio >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "created wallet: eosio."
  else
      failure 'failed to create eosio wallet.'
  fi
}

function import_eosio() {
  cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "imported eosio key."
  else
      failure 'failed to import eosio key.'
  fi
}

function unlock_eosio() {
  cleos -u http://0.0.0.0:8888 wallet unlock -n eosio --password $EOSIO_PASSWORD >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "unlocked eosio."
  else
      success 'unlocked eosio.'
  fi
}

function create_account() {
  cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "created account: $IDENTITY."
  else
      failure "failed to created account: $IDENTITY."
  fi
}

function publish_account() {
  TRANSACTION="["$IDENTITY", "$ADDRESS", "$PORT"]"
  cleos -u $EOSURL push action bittensoracc upsert "$TRANSACTION" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1

  if [ $? -eq 0 ]; then
      success "published account: $IDENTITY."
  else
      failure "failed to published account: $IDENTITY."
  fi
}

function unpublish_account() {
  TRANSACTION="["$IDENTITY"]"
  cleos -u $EOSURL push action bittensoracc erase "$TRANSACTION" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      success "unpublished account: $IDENTITY."
  else
      failure "failed to unpublish account: $IDENTITY."
  fi
}

log "=== BitTensor ==="
log "Args {"
log "   EOSURL: $EOSURL"
log "   IDENTITY: $IDENTITY"
log "   ADDRESS: $ADDRESS"
log "   PORT: $PORT"
log "}"

# Make state folder and logs file.
mkdir data/$IDENTITY
touch data/$IDENTITY/bittensor_logs.out

log ""
log "=== setup accounts ==="

# Check to see if eosio wallet exists.
# If not, create eosio account and pull private keys to this wallet.
PUBLIC_KEY=$(cleos -u $EOSURL wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')
EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
EOSIO_PUBLIC_KEY=EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV
EOSIO_PASSWORD=PW5JgJBjC1QXf8XoYPFY56qF5SJLLJNfjHbCai3DyC6We1FeBRL8q

# Create Wallet if does not exist.
if [[ $EOSIO_PUBLIC_KEY != $PUBLIC_KEY ]]; then
  create_eosio
  import_eosio
fi

unlock_eosio

create_account

publish_account

# Final call to clean account.
trap unpublish_account EXIT

log ""
log "=== start neuron ==="
python src/main.py $IDENTITY $ADDRESS $PORT $EOSURL
log "neuron shut down."
