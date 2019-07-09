## BitTensor Network Daemon

<img src="assets/mycellium.jpeg">

> Decentralized Machine Intelligence

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Incentive Structure](#incentivestructure)
- [To Run](#torun)
- [What is the state of this project](#project-state)
- [About Word Embeddings](#word-embeddings)
- [License](#license)

---

## Overview

BitTensor is a tool for building decentralized Machine Intelligence systems. When run, this software is built to fold your computing power into an p2p machine learning system. The network is aligned by an incentive model built on EOS, which uses a recommendation system to reward computers for their informational product.

## Motivation

Machine intelligence has been successfully mapped onto both distributed and [hardware](https://knowm.org/) systems, but has not yet entered a decentralized environment. This setting has been harnessed by other technologies to bring considerable volumes of [computing power](https://digiconomist.net/bitcoin-energy-consumption) and a large and diverse number of [collaborators](https://en.wikipedia.org/wiki/BitTorrent) to bear on a problem domain. This is promising for Machine Learning in particular which requires large amounts of computing power and benefits from extending model [capacity](https://arxiv.org/abs/1701.06538), [diversity](https://arxiv.org/pdf/1611.05725.pdf), and [collaboration](https://en.wikipedia.org/wiki/Ensemble_learning).

Further more, these benefits are no stranger to intelligent systems in nature --such as the neural networks, societies at large, or plant structures like mycelium-- which run through the interaction of many self-interested parts rather than under centralized executive authority.

## Incentive Structure     

The BitTensor meta-machine learning model is composed of a many interconnected machines, each running their own sub-machine learning models. The connections between these nodes reflect channels along with Tensors are passed containing, in the forward direction features, and in the reverse direction gradients. Client nodes communicate upstream to servers and while training their own, subjective objective function, produce attribution values, which are numerical evaluations of each connection. The manner in which these attributions are calculates is arbitrary, for instance, by calculating Fishers Information Metric.

Attributions are posted discretely to the EOS blockchain and in total, construct a directed weighted graph (DWG) structure. See Figure 1, bellow for an example. Given the DWG we are able to calculate the attribution scores between the network and each node in the graph to produce a node ranking. The contract then emits newly minted tokens at a constant rate to nodes, in proportion to their value to the metagraph.

<img src="assets/weboftrust.jpg">

The calculation for this can be viewed in python numpy format, with matrix multiplication in the file "Emission testing.ipynb". The actual emission system is an approxiation to this matrix based calculation, but instead splits the computation across each of the networks nodes. A python implementation can also be viewed in "Emission testing.ipynb", with the EOS c++ code writing in bittensor.cpp.

As is, the emission scheme is linear. Each EOS block emits 1 token to the network which is split between each member node in accordance to that node's overall network attribution.

For instance, the graph structure represented by the following values:

in_edge_weights = [0.6 0.9 0.4 0.5 0.5  0.5 1. 1.  1.  1. ]
initial_stake = [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
adjacency_weights =
[[0. 0.1 0.3 0.2 0.  0.  0.  0.  0.  0. ],
[0.1 0.  0.  0.1 0.  0.  0.  0.  0.  0. ],
[0.1 0.  0.  0.2 0.  0.  0.  0.  0.  0. ],
[0.2 0.  0.3 0.  0.  0.  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.  0.5  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.5  0.  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ],
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]

Give the following attributions:
Attributions: 0.04884 0.30159 0.01609 0.03348 0.10000 0.10000 0.10000 0.10000 0.10000 0.10000

and the following Emissions after a single block.
Emission: 0.04884 0.30159 0.01609 0.03348 0.10000 0.10000 0.10000 0.10000 0.10000 0.10000  --> sum = 1.0

---

## To Run

Required:
1. [Docker](https://docs.docker.com/install/)
1. [Python3.7](https://realpython.com/installing-python/)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

$ pip install -r requirements.txt

# Run EOS blockchain.
$ ./start_blockcahin.sh

# Run Node 1.
# ./start_bittensor.sh

# Run Node 2.
# ./start_bittensor.sh

...

# Run Node N.
# ./start_bittensor.sh

```



# About Word Embeddings

A word embedding is a projection from a word to a continuous vector space 'cat' --> [0,1, 0,9, ..., -1.2], which attempts to maintain the word's semantics. For instance, 'King' - 'Queen' = 'Male'. Word embeddings are highly useful first order projections for a number of Machine Learning problems which make them an ideal product for a network attempting to be useful for the largest number of individuals.

Word embeddings can be trained in a self-supervised manner on a language corpus by attempting to find a projection which helps a classifier predict words in context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or ('hair' -> 'long'). The ability to predict context requires an understanding of the relationships between words ('meaning' is relational quality). The assumption is that the meaning of a word is determined by the company it keeps been highly successful assumption in practice.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding producing method is possible. The goal is diversity.

## License

MIT
