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

Computing within a decentralized environment is necessarily more difficult than a traditional system since individual components may be non-existent, incompetent, or malicious. Examples like Bitcoin solve this problem by constructing a game theoretic structure governed by a protocol which rewards proper contribution, in this case, raw hashing power.

A Machine Learning system requires a different form a contribution. Instead, BitTensor rewards the production of useful information, encoded as tensor projections, for instance a word embedding. Value determined by a web-of-trust like recommendation system where recommendations are based on attribution methods run on each host - ranking their neighbors using an information theoretic metric called Fishers Information. In total, these rankings, paired with stake, determine the likelihood of receiving emission rewards and this forms the incentive model.

<img src="assets/weboftrust.jpg">

A note on the proper determination of use. Informational significance (determined by an attribution method) is calculated on a pair wise basis, from one node to another, in a subjective manner. However, subjectivity allows network nodes to falsify their attribution in order to gain a higher emission rewards (for themselves). 

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
