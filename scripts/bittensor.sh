#!/usr/bin/env bash
# change to script's directory
IDENTITY=$1
ADDRESS=$2
PORT=$3
EOSURL=$4

echo "EOSURL: " $EOSURL
echo "IDENTITY: " $IDENTITY
echo "ADDRESS: " $ADDRESS
echo "PORT: " $PORT

echo "=== run bittensor ==="

echo -e "=== setup wallet: eosio ==="
# First key import is for eosio system account
EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
EOSIO_PUBLIC_KEY=EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV

# Check to see if eosio wallet exists
PUBLIC_KEY=$(cleos -u $EOSURL wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')

if [[ $EOSIO_PUBLIC_KEY != $PUBLIC_KEY ]]; then

  echo -e "cleos -u $EOSURL wallet create -n eosio --to-console"
  cleos -u $EOSURL wallet create -n eosio

  echo -e "cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY"
  cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY

fi

echo -e "=== create account ==="
echo -e "cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY"
cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY
echo -e "=== done create account ==="
echo -e ''

echo -e "=== publish account: $ADDRESS ==="
TRANSACTION="["$IDENTITY", "$ADDRESS", "$PORT"]"
echo $TRANSACTION
echo -e "cleos -u $EOSURL push action bittensoracc upsert $TRANSACTION -p $ACCOUNT_NAME@active"
cleos -u $EOSURL push action bittensoracc upsert "$TRANSACTION" -p $IDENTITY@active
echo -e "=== done publishing peer address ==="
echo -e ''

echo -e "=== start neuron: $IDENTITY ==="
mkdir checkpoints
mkdir checkpoints/$IDENTITY
python src/main.py $IDENTITY $ADDRESS $PORT $EOSURL

echo -e "=== erase account: $IDENTITY ==="
TRANSACTION="["$IDENTITY"]"
echo $TRANSACTION
echo -e "cleos -u $EOSURL push action bittensoracc erase $TRANSACTION -p $IDENTITY@active"
cleos -u $EOSURL push action bittensoracc erase "$TRANSACTION" -p $IDENTITY@active
echo -e "=== done publishing peer address ==="
echo -e ''
