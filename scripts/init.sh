echo "EOSURL: " $EOSURL
echo "IDENTITY: " $IDENTITY
echo "ADDRESS: " $ADDRESS
echo "PORT: " $PORT

# Build account on the EOS blockchain.
scripts/init_account.sh

# Start Neuron training.
scripts/init_neuron.sh
