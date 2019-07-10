## BitTensor Network Daemon

<img src="assets/mycellium.jpeg">

> Decentralized Machine Intelligence

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Organization](#organization)
  - [Nucleus](#nucleus)
  - [Dendrite](#dendrite)
  - [Synapse](#synapse)
  - [Metagraph](#metagraph)
- [Incentive](#incentive)
- [To-Run](#to-run)
- [Word-Embeddings](#word-embeddings)
- [License](#license)

---

## Overview

BitTensor is a Machine Intelligence system which enables any computer in the world the ability to share, and profit from, its ability to learn.

When this software is run, it is built to fold your computing power into a single p2p machine learning network. The network is aligned by an incentive model built on EOS, which uses a recommendation system to reward computers for their informational product -- and pay accordingly.

## Motivation

Machine intelligence has been successfully mapped onto both distributed and [hardware](https://knowm.org/) systems, but has not yet entered a decentralized environment. This setting has been harnessed by other technologies to bring considerable volumes of [computing power](https://digiconomist.net/bitcoin-energy-consumption) and a large and diverse number of [collaborators](https://en.wikipedia.org/wiki/BitTorrent) to bear on a problem domain. This is promising for Machine Learning in particular which requires large amounts of computing power and benefits from extending model [capacity](https://arxiv.org/abs/1701.06538), [diversity](https://arxiv.org/pdf/1611.05725.pdf), and [collaboration](https://en.wikipedia.org/wiki/Ensemble_learning).

These benefits are no stranger to intelligent systems in nature --such as the neural networks, societies at large, or plant structures like mycelium-- which run through the interaction of many self-interested parts rather than under centralized executive authority.

There is a good reason for this: a centralized system requires (by definition) the aggregation of information, but for highly complex systems, like the brain, this is prohibitive. A system which can maintain organization without a central coordinator is capable of greater complexity, and diversity.

The key to building such a system is the development of a suitable protocol, language of value, or culture, which aligns the components of the system. Assumptions about the self-interest of individual components may be enough to keep the system working as intended, as is the case with Bitcoin, without executive overhead.

BitTensor uses an incentive model organized around a token emission scheme and recommendation network. (Explained under 'Incentive') The token we use, BDNF, is biologically inspired by its neurotransmitter namesake, which acts in the human central nervous system to stimulate neurogenesis and survival: 'The brain is a market, and BDNF is its currency.'

## Organization

```
                 [EOS]
                   |
              [Metagraph]
           /       |       \
[Dendrite] ---> [Nucleus] ---> [Synapse]
           \       |       /
                 [Main]
```

###### Nucleus
The main Tensorflow graph is defined and trained within the Nucleus object. As is, the class is training a self supervised word embedding over a dummy corpus of sentences in text8.zip. The result is a mapping which takes word to a 128 dimension vector, representing that word while maintaining its semantic properties.

Although subject to future change, this problem serves as a good starting place because its generality and ubiquity within Artificial intelligence. In future versions of this code, this will be expanded to include sentence and paragraph embeddings, speech, image and video embeddings with the goal of training the network for general multitask.

###### Dendrite
During training the Nucleus interacts with the rest of the network through its Dendrite. The Dendrite maintains connections to upstream nodes making asynchronous calls using GRPC, and passing serialized Tensor protocol buffers along the wire.

During validation and inference the Dendrite is cut from the model and replaced by submodules which have been trained through distillation to approximate the incoming signals from the rest of the network.

###### Synapse
This inference graphs being produced in training are served by the Synapse object. The Synapse is responsible for upstream connections. It is responsible for rate limiting, and through this,  negotiating for higher attribution within the Metagraph.

Since the Synapse object is merely serving the inference graph, it is mostly detached from the Nucleus and Dendrite during training, only communicating with these objects by pulling the latest and best inference graph from the storage directory.

###### Metagraph
The Metagraph object acts as an interface between the EOS blockchain and the rest of the neuron. Through the Metagraph, this node can post updated attributions and call timed token emission (which releases newly mined tokens) The Metagraph object also serves as a de-facto DHT which removes the need for a gossip protocol used by many standard p2p applications Bitcoin and BitTorrent not withstanding.

###### EOS
The EOS contract is separate from Dendrite. Nucleus, Synapse and Metagraph objects during execution. During testing, this class is run on a local EOS instance, but during production the contract is running in a decentralized manner across the EOS network.  


## Incentive     

The BitTensor network, in aggregate, forms a single meta machine learning model composed of a many interconnected nodes, each running their own sub machine learning models. The connections between these nodes reflect channels along which Tensors are passed, containing, in the forward direction, features, and in the reverse direction gradients: No different than the individual layers of a standard Neural Network architecture (or Tensorflow graph)

<img src="assets/NN.jpeg", width="600">

Client nodes communicate upstream to servers and, while training their local objective function, produce attribution values, which are numerical evaluations of each connection. The manner in which these attributions are calculatef is arbitrary (for instance, by calculating Fishers Information Metric), but
the result is a directed weighted graph (DWG) structure. (Below)

<img src="assets/weboftrust.jpg">

The DWG is updated discretely through emit transactions, and are conglomerated on the EOS blockchain. In total, we are able to calculate a further attribution scores, between the network and each node, which reflects each node's global ranking.

Our Token emission scheme is designed around these global rankings so that newly minted tokens are distributed in proportion to each node's value in the metagraph. The emission calculation in python-numpy format is seen bellow:

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

## To-Run

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

## Word-Embeddings

A word embedding is a projection from a word to a continuous vector space 'cat' --> [0,1, 0,9, ..., -1.2], which attempts to maintain the word's semantics. For instance, 'King' - 'Queen' = 'Male'. Word embeddings are highly useful first order projections for a number of Machine Learning problems which make them an ideal product for a network attempting to be useful for the largest number of individuals.

Word embeddings can be trained in a self-supervised manner on a language corpus by attempting to find a projection which helps a classifier predict words in context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or ('hair' -> 'long'). The ability to predict context requires an understanding of the relationships between words ('meaning' is relational quality). The assumption is that the meaning of a word is determined by the company it keeps been highly successful assumption in practice.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding producing method is possible. The goal is diversity.

## License

MIT
