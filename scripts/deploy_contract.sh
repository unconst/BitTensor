#!/usr/bin/env bash
set -o errexit
source scripts/constant.sh

# $1 smart contract name
# $2 account holder name of the smart contract
# $3 wallet for unlocking the account
# $4 password for unlocking the wallet

# set PATH
PATH="$PATH:/opt/eosio/bin"

CONTRACTSPATH="$( pwd -P )/contract"

# make new directory for compiled contract files
mkdir -p ./compiled_contracts
mkdir -p ./compiled_contracts/$1

COMPILEDCONTRACTSPATH="$( pwd -P )/compiled_contracts"

# unlock the wallet, ignore error if already unlocked
if [ ! -z $3 ]; then cleos wallet unlock -n $3 --password $4 || true; fi

# compile smart contract to wasm and abi files using EOSIO.CDT (Contract Development Toolkit)
# https://github.com/EOSIO/eosio.cdt
log "=== compile contract: $1 ==="
eosio-cpp -abigen "$CONTRACTSPATH/$1/$1.cpp" -o "$COMPILEDCONTRACTSPATH/$1/$1.wasm" --contract "$1"

# set (deploy) compiled contract to blockchain
log "=== cleos set contract: $1 ==="
cleos set contract $2 "$COMPILEDCONTRACTSPATH/$1/" --permission $2
