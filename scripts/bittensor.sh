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

# Make state folder and logs file.
mkdir data/$IDENTITY
touch data/$IDENTITY/bittensor_logs.out

echo "=== run bittensor ==="

# Check to see if eosio wallet exists.
# If not, create eosio account and pull private keys to this wallet.
PUBLIC_KEY=$(cleos -u $EOSURL wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')
EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
EOSIO_PUBLIC_KEY=EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV
if [[ $EOSIO_PUBLIC_KEY != $PUBLIC_KEY ]]; then

  echo -e "=== setup wallet: eosio ==="
  echo -e "cleos -u $EOSURL wallet create -n eosio --to-console"
  cleos -u $EOSURL wallet create -n eosio

  echo -e "cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY"
  cleos -u $EOSURL wallet import -n eosio --private-key $EOSIO_PRIVATE_KEY

fi

# Create our owned account to the EOS chain.
echo -e "=== create account: $IDENTITY ==="
echo -e "cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY"
cleos -u $EOSURL create account eosio $IDENTITY $EOSIO_PUBLIC_KEY $EOSIO_PUBLIC_KEY >> data/$IDENTITY/bittensor_logs.out 2>&1

if [ $? -eq 0 ]; then
    echo "Successfuly created account: $IDENTITY"
else
    echo 'Failed to EOS account.'
fi
echo -e ''

# Publish our account to the Bittensor Contract.
echo -e "=== publish account: $IDENTITY ==="
TRANSACTION="["$IDENTITY", "$ADDRESS", "$PORT"]"
echo -e "cleos -u $EOSURL push action bittensoracc upsert $TRANSACTION -p $ACCOUNT_NAME@active"
cleos -u $EOSURL push action bittensoracc upsert "$TRANSACTION" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1

if [ $? -eq 0 ]; then
    echo "Successfuly published account: $IDENTITY."
else
    echo 'Failed to published account.'
fi
echo -e ''


# Deletes the stale account after shutdown.
function delete_account {
  echo -e "=== erase account: $IDENTITY ==="
  TRANSACTION="["$IDENTITY"]"
  echo $TRANSACTION
  echo -e "cleos -u $EOSURL push action bittensoracc erase $TRANSACTION -p $IDENTITY@active"
  cleos -u $EOSURL push action bittensoracc erase "$TRANSACTION" -p $IDENTITY@active >> data/$IDENTITY/bittensor_logs.out 2>&1
  if [ $? -eq 0 ]; then
      echo "Successfuly cleared account: $IDENTITY."
  else
      echo 'Failed to clear account.'
  fi
  echo -e ''
}

# Final call to clean account.
trap delete_account EXIT

echo -e "=== start neuron: $IDENTITY ==="
python src/main.py $IDENTITY $ADDRESS $PORT $EOSURL
