## BitTensor Network Daemon

<img src="mycellium.jpeg">

> Unsupervised Incentivized Peer-to-Peer Machine Learning.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [How to run](#how-to-run)
- [What is the state of this project](#project-state)
- [About Word Embeddings](#word-embeddings)
- [License](#license)

## Overview

This is a research tool for working with unsuperivsed peer to peer Machine Learning systems. The goals is to find novel ways of combining computing
resources which extend beyond a trusted computing boundary. The hope is that incentivization models like the ones in Bitcoin or BitTorrent can be used to
align the computing resources and keep them online and working towards a worthy Machine Learning goal. An unsupervised task seems ideal in this environemnt
because data is easy to come by and each computer can work on their own without the need for a supervised moderator.

# How to run

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

```

# What is the state of this project?

The server code spins up a Neuron object which is training a standard word embedding. This NN trains over a simple dataset in text8.zip to produce a 128 dim
word embedding which it exposed on a GRPC server. Currently, tracking and metagrah construction are not implemented, nor is an incentive structure which I
hope will center around a easily transactable crypto currency. As is, you can specify another server on the local host and your node will combine the trained
embedding from that of your own during training.

# About Word Embeddings.

A word embedding is a projection from a word to a continuous vector space 'cat' --> [0,1, 0,9, ..., -1.2], which attempts to maintain the word's semantics.
For instance, 'King' - 'Queen' = 'Male'. Word embeddings are highly useful first order projections for a number of Machine Learning problems which makes
them an ideal product for a p2p machine learning network attempting to be useful for the largest number of individuals.

Word embeddings can be trained in a self-supervised manner on a language corpus by attempting to find a projection which helps a classifier predict word in
it's context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or
('hair' -> 'long'). The assumption is that the meaning of a word is determined by the company it keeps. In practice this assumption has been highly succesful.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding
producing method is possible. The goal is diversity.

## License

MIT
