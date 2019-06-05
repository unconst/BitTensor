

echo -e "=== setup wallet: eosiomain ==="
# First key import is for eosio system account
EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
echo -e "cleos -u $EOSURL wallet create -n eosiomain --to-console"
cleos -u $EOSURL wallet create -n eosiomain --to-console | tail -1 | sed -e 's/^"//' -e 's/"$//' > eosiomain_wallet_password.txt

echo -e "cleos -u $EOSURL wallet import -n eosiomain --private-key $EOSIO_PRIVATE_KEY"
cleos -u $EOSURL wallet import -n eosiomain --private-key $EOSIO_PRIVATE_KEY

echo -e "=== done setup wallet: eosiomain ==="
echo -e ''

echo -e "=== setup wallet: $IDENTITY ==="
echo -e "cleos -u $EOSURL wallet create --name $IDENTITY --file $IDENTITY-wallet.txt"
cleos -u $EOSURL wallet create --name $IDENTITY --file "$IDENTITY"-wallet.txt
echo -e "=== done setup wallet: $IDENTITY ==="
echo -e ''

echo -e "=== get public key for: $IDENTITY ==="
echo -e "cleos -u $EOSURL wallet key"
PUBLIC_KEY=$(cleos -u $EOSURL wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')
echo "PUBLIC Key is:" $PUBLIC_KEY
echo -e "=== done create wallet ==="
echo -e ''

echo -e "=== create account ==="
ACCOUNT_NAME="$IDENTITY"
echo -e "cleos -u $EOSURL create account eosio $ACCOUNT_NAME $PUBLIC_KEY $PUBLIC_KEY"
cleos -u $EOSURL create account eosio $ACCOUNT_NAME $PUBLIC_KEY $PUBLIC_KEY
echo -e "=== done create account ==="
echo -e ''

echo -e "=== publish peer address: $ADDRESS ==="
TRANSACTION="["$ACCOUNT_NAME", "$ADDRESS", "$PORT"]"
echo $TRANSACTION
echo -e "cleos -u $EOSURL push action bittensoracc upsert $TRANSACTION -p $ACCOUNT_NAME@active"
cleos -u $EOSURL push action bittensoracc upsert "$TRANSACTION" -p $ACCOUNT_NAME@active
echo -e "=== done publishing peer address ==="
echo -e ''

echo -e "=== pull metagraph ==="
echo -e "cleos -u $EOSURL get table bittensoracc bittensoracc peers"
cleos -u $EOSURL get table bittensoracc bittensoracc peers
echo -e "=== done pull metagraph ==="
echo -e ''
