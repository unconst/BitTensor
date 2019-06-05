

echo -e "=== start neuron: $IDENTITY ==="
mkdir checkpoints
mkdir checkpoints/$IDENTITY
python src/main.py $IDENTITY $ADDRESS $PORT $EOSURL
