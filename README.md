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

A method of training computers connected across the internet, instead of within a closed environment, could allow us to build Machine Intelligence systems with hitherto unseen size and complexity. 

We propose BitTensor as new class of Machine Learning model to do this, one which trains across a peer-to-peer network and enables any computer and any engineer in the world to contribute to its training. 

The nature of trust-less computing necessitates that these contributions are driven by incentives rather than by direct control from any one computer. We use a digital token to carry that incentive signal through the network where its magnitude is a function of informational significance. 

In addition, BitTensor uses local-learning techniques to remove the need for centralized coordination, as such it remains robust and efficient during both horizontal and vertical scaling across the web.

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
