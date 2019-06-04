

IDENTITY=$1
ADDRESS=$2


echo -e "=== setup wallet: eosiomain ==="
# First key import is for eosio system account
EOSIO_PRIVATE_KEY=5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3
cleos -u http://localhost:8888 wallet create -n eosiomain --to-console | tail -1 | sed -e 's/^"//' -e 's/"$//' > eosiomain_wallet_password.txt
cleos -u http://localhost:8888 wallet import -n eosiomain --private-key $EOSIO_PRIVATE_KEY
echo -e "=== done setup wallet: eosiomain ==="
echo -e ''

echo -e "=== setup wallet: $IDENTITY ==="
cleos -u http://localhost:8888 wallet create --name $IDENTITY --file "$IDENTITY"-wallet.txt
echo -e "=== done setup wallet: $IDENTITY ==="
echo -e ''

echo -e "=== get public key for: $IDENTITY ==="
PUBLIC_KEY=$(cleos -u http://localhost:8888 wallet keys | tail -2 | head -n 1 | tr -d '"' | tr -d ' ')
echo "PUBLIC Key is:" $PUBLIC_KEY
echo -e "=== done create wallet ==="
echo -e ''

echo -e "=== create account ==="
ACCOUNT_NAME="$IDENTITY"acc
cleos -u http://localhost:8888 create account eosio $ACCOUNT_NAME $PUBLIC_KEY $PUBLIC_KEY
echo -e "=== done create account ==="
echo -e ''

echo -e "=== publish peer address: $ADDRESS ==="
TRANSACTION="["$ACCOUNT_NAME", "$ADDRESS"]"
echo $TRANSACTION
cleos -u http://localhost:8888 push action bittensoracc upsert "$TRANSACTION" -p $ACCOUNT_NAME@active
echo -e "=== done publishing peer address ==="
echo -e ''

echo -e "=== pull metagraph ==="
cleos -u http://localhost:8888 get table bittensoracc bittensoracc peers
echo -e "=== done pull metagraph ==="
echo -e ''
