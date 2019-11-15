## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

## Table of Contents

- [Overview](#overview)
- [To-Run](#to-run)
- [Neurons](#neurons)
- [License](#license)

---

## Overview

A machine intelligence trained with access to the internet could harness resources of data, machine knowledge and computer power untapped by its predecessors.

This repo contains a protocol implementation for a trustless, decentralized and incentivised peer-to-peer model that can connect an unlimited number of computers and engineers across the web.

In absence of centralized control, the network uses a digital token to provide incentive for intelligence contribution. The reciever has more sway over the network's mathematical objective.

Each learning module use (1) delayed gradients and (2) synthetic inputs to cut independence between nodes in the network, and (3) Fishers Information pruning to evaluate the contribution of our neighbors.

Version 1 focuses on language embeddings but the final result will focus on general multitask learning.

## To-Run

Running this software creates and connects your instance into the network. To begin, you will need to:

1. [Install Docker](https://docs.docker.com/install/)
1. [Make account on Digital Ocean](https://www.digitalocean.com/)
1. [Make a Digital Ocean API key](https://cloud.digitalocean.com/account/api/tokens)

Then:

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run a Remote Node
$ ./bittensor.sh --remote --token $DIGITAL_OCEAN_TOKEN --eosurl http://142.93.177.245:8888
```

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

## Neurons

1. Mach: General purpose neuron which learns a language embedding, shares this with its neighbors.

1. Boltzmann: Trainable Feed-forward NN, tokenizes requests on a word level basis responds with 128 dimensional embedding vectors. Applies gradients on 3 second interval without averages.

1. Feynmann: Trains a embedding model over the network using a dummy corpus from text.zip. Serves trained model to network. Does not apply gradients.

1. Elmo: Untrainable Elmo NN.

1. GoogleUSE: Google's universal sentence encoder. Non-trainable. Previously trained on question and answer text.

1. CoLA: CoLA dataset node, learning to classify speech.

## License

MIT
