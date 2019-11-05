## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

## Table of Contents

- [Overview](#overview)
- [To-Run-Locally](#to-run-locally)
- [To-Run-Testnet](#to-run-testnet)
- [To-Run-CoLA](#to-run-cola)
- [Neurons](#neurons)
- [License](#license)

---

## Overview

A machine intelligence trained with access to the internet could unify compute, machine knowledge, and data, currently untapped by its predecessors. 

This repository contains a peer-to-peer machine learning model that combines an unbounded number of computers across the web. 

The nature of trust-less computing necessitates that these computers are driven by incentives rather than direct control from any one computer. 

This software uses a digital token to carry that incentive signal through the network. Its magnitude is a function of the informational significance of the receiving computer.

The learning modules use two techniques (1) delayed gradients and (2) synthetic inputs to make the system robust and efficient. 

Running the software connects your instance to this machine learning model and rewards you for producing information significant in reducing the overall error in the network.

## To-Run-Locally
1. [Install Docker](https://docs.docker.com/install/)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run a test EOS blockchain.
$ ./start_eos.sh

# Run Node 1. A pretrained ELMO model.
$ ./bittensor.sh --neuron ELMO

# Run Node 2. An intermediate node which learns from its children.
$ ./bittensor.sh --neuron Boltzmann

...

# Run Node N. A component training against the CoLA text classification dataset.
$ ./bittensor.sh --neuron CoLA

```

## To-Run-Testnet

1. [Install Docker](https://docs.docker.com/install/)
1. [Make account on Digital Ocean](https://www.digitalocean.com/)
1. [Make a Digital Ocean API key](https://cloud.digitalocean.com/account/api/tokens)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run a Remote Node
$ ./bittensor.sh --remote --token $DIGITAL_OCEAN_TOKEN --eosurl http://142.93.177.245:8888


# Run a Local Node
$ python src/upncp.py --port 9091  // To punch a hole in your router.
$ ./bittensor.sh --port 9091 --eosurl http://142.93.177.245:8888

```

## To-Run-CoLA
1. [Install Docker](https://docs.docker.com/install/)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run a test EOS blockchain.
$ ./start_eos.sh

# Run Node 1.
$ ./bittensor.sh --neuron ELMO

# Run Node 2.
$ ./bittensor.sh --neuron CoLA

```

## Pull Requests

We use [Yapf](https://github.com/google/yapf) for code format. Please run the following.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

## Neurons

1. Boltzmann: Trainable Feed-forward NN, tokenizes requests on a word level basis responds with 128 dimensional embedding vectors. Applies gradients on 3 second interval without averages.

1. Feynmann: Trains a embedding model over the network using a dummy corpus from text.zip. Serves trained model to network. Does not apply gradients.

1. Elmo: Untrainable Elmo NN.

1. GoogleUSE: Google's universal sentence encoder. Non-trainable. Trained on question and answer text.

1. CoLA: CoLA dataset node, learning to classify speech. 

## Further Reading

The system description can be found in docs/Bittensor_Biological_Scale_NeuralNetworks


## License

MIT
