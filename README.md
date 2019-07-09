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

## Daemon Organization

```
                 [EOS]
                   |
              [Metagraph]
           /       |       \
[Dendrite] ---> [Soma] ---> [Synapse]
           \       |       /
                [Main]
```

###### SOMA
The main Tensorflow graph is defined and trained within the Soma object. As is, this class is training a self supervised word embedding over a dummy corpus of sentences in text8.zip. The result is a mapping which takes word to a 128 dimension vector, representing that word, and maintaining its semantic properties. This problem serves as a good starting place because of word embeddings [wide use] and generality in Artificial intelligence. In future versions of this code, this problems solved with be expanded to include sentence and paragraph embeddings, speech, image and video embeddings with the goal of training the network for general multitask.

###### DENDRITE
During training the Soma interacts with the rest of the network through its Dendrite, which maintains connections to upstream nodes. The Dendrite makes asynchronous calls using GRPC, passing protocol buffer serialized Tensors. During validation and inference the Dendrite is cut from the model and replaced by submodules which have been trained through distillation to approximate the incoming signals from the rest of the network.

###### SYNAPSE
This inference graphs being produced in training are served by the Synapse object. The Synapse is responsible for upstream connections. It is responsible for rate limiting, and through this,  negotiating for higher attribution within the Metagraph. Since the Synapse object is merely serving the inference graph, it is mostly detached from the Soma and Dendrite during training, only communicating with these objects by pulling the latest and best inference graph from the storage directory.

###### METAGRAPH
The Metagraph object acts an interface between the EOS blockchain and the rest of the neuron. Through the Metagraph, this node can post updated attributions and call timed token emission (which releases newly mined tokens) The Metagraph object also serves as a de-facto DHT which removes the need for a gossip protocol used by many standard p2p applications Bitcoin and BitTorrent not withstanding.

###### EOS
The EOS contract is separate from Dendrite. Soma, Synapse and Metagraph objects during execution. During testing, this class is run on a local EOS instance, but during production the contract is running in a decentralized manner across the EOS network.  


## Incentive Structure     

The BitTensor network, in aggregate, forms a single meta machine learning model composed of a many interconnected nodes, each running their own sub machine learning models. The connections between these nodes reflect channels along which Tensors are passed, containing, in the forward direction, features, and in the reverse direction gradients: No different than the individual layers of a standard Neural Network architecture.

Client nodes communicate upstream to servers and, while training their local objective function, produce attribution values, which are numerical evaluations of each connection. The manner in which these attributions are calculates is arbitrary (for instance, by calculating Fishers Information Metric), but
the result is a directed weighted graph (DWG) structure. See Figure 1, bellow for an example.

<img src="assets/weboftrust.jpg">

The DWG is updated discretely through emit transactions, and are conglomerated on the EOS blockchain. In total, we are able to calculate a further attribution scores, between the network and each node, which reflects each node's global ranking.

Our Token emission scheme is designed around these global rankings so that newly minted tokens are distributed in proportion to each node's value in the metagraph.

The emission calculation in python-numpy format is seen bellow:

```
def bittensor_emission_simulation():

    # Stake vector.
    S = [1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

    # Loop-in edges.
    N = [0.6 0.9 0.4 0.5 0.5 0.5  1.  1.  1.  1. ]

    # Outgoing edges.
    M =[[0.  0.1 0.3 0.2 0.  0.  0.  0.  0.  0. ]
        [0.1 0.  0.  0.1 0.  0.  0.  0.  0.  0. ]
        [0.1 0.  0.  0.2 0.  0.  0.  0.  0.  0. ]
        [0.2 0.  0.3 0.  0.  0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.  0.5 0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.5 0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
        [0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]

    # Loop over blocks.
    n_blocks = 100
    for _ in range(n_blocks):        

        # Attribution calculation.
        depth = 100
        A = np.multiply(S, N)
        T = np.matmul(M, S)
        for _ in range(depth):
            A += np.multiply(T, N)
            T = np.matmul(M, T)

        # Emission calculation.
        tokens_per_block = 50
        A = A / np.linalg.norm(A, 1)
        E = A * tokens_per_block
        S = S + E
```


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
