## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

> Decentralized Machine Intelligence

## Table of Contents

- [Overview](#overview)
- [To-Run-Locally](#to-run-locally)
- [To-Run-Testnet](#to-run-testnet)
- [Why](#why)
- [How](#how)
- [Organization](#organization)
  - [Nucleus](#nucleus)
  - [Dendrite](#dendrite)
  - [Synapse](#synapse)
  - [Metagraph](#metagraph)
- [Incentive](#incentive)
- [Word-Embeddings](#word-embeddings)
- [License](#license)

---

## Overview

BitTensor is a new class of Machine Learning model which is trained across a peer-to-peer network. It enables any computer and any engineer in the world to contribute in training.

The nature of trust-less computing necessitates that these contributions are combined through incentive rather than direct control from any one computer. We use a digital token to carry that incentive signal through the network: where the magnitude of this incentive is derived from a p2p collaborative filtering technique similar to Google's Page rank algorithm.  

The lack of centrality allows the structure to grow to arbitrary size across the internet. Both the cost and control of the system is distributed. And the network's informational product is priced into the reward token's value.

When run, this software folds your computing power into a p2p network and rewards you with an EOS based digital token for your contribution.

## To-Run-Locally
1. [Install Docker](https://docs.docker.com/install/)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run EOS blockchain.
$ ./start_eos.sh

# Run Node 1.
$ ./bittensor.sh

# Run Node 2.
$ ./bittensor.sh

...

# Run Node N.
$ ./bittensor.sh

```

## To-Run-Testnet

1. [Install Docker](https://docs.docker.com/install/)
1. [Make account on Digital Ocean](https://www.digitalocean.com/)
1. [Make a Digital Ocean API key](https://cloud.digitalocean.com/account/api/tokens)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run Remote Node
$ ./bittensor.sh --remote --token $DIGITAL_OCEAN_TOKEN --eosurl http://142.93.177.245:8888


# Run Local node
$ python src/upncp.py --port 9091  // To punch a hole in your router.
$ ./bittensor.sh --port 9091 --eosurl http://142.93.177.245:8888

```

<img src="assets/brain.png" width="1000" />

## Why

This technology is being built because we believe the production of Machine Intelligence, like Human Intelligence, advances understanding and understanding begets harmony. And yet, intelligence is power and power, if held in the hands of the few, will corrupt. Because of this, our technological approach attempts to democratize ownership and opens its value stream to any computer and any individual who deems it worthwhile to contribute.

Moreover, although democratization and openness are ethical values, we are relying on their practical use here: A free and open system with a large number of stake holders is also the most direct path towards our goal of producing Strong Machine Intelligence. The scale of the AI problem in front of us necessitates that we build it this way.

Why is this? Because a decentralized computing approach harnesses the largest pool of computing power and the largest pool of collaborators: Any computer and any engineer can contribute to this system.

We've seen from this technology's predecessors, Bitcoin and BitTorrent the power open source systems can unlock: At their zenith, these two technologies were, respectively, the largest super computer and largest bandwidth user across the globe.

<img src="assets/Lightning.png" width="1000" />

Above: Bitcoin Lightning network nodes from late 2018.

## How

In standard Machine learning setting, the training mechanism uses Back-propagation to minimize the loss on a dataset with respect to the weights, and at each step the model parameters must wait for a signal from the loss function before the parameters can be updated.

This is prohibitive when the scale of those networks reach the scale desired by modern machine learning needs -- or biological scale -- and necessarily so when we are attempting to train a system which spans multiple computers connected across the web, as we are doing here.

Training networks composed of many 'local' loss functions allow us to train subsections of the network independently, dividing and conquering the problem so that each locality is not (immediately) dependent on far off events in the large network. This is not dissimilar to the decentralized/parallel structure of the human brain, and has been successfully applied to increase the scale of Neural Networks into the trillion parameter range. [Gomez 2019].

<img src="assets/kgraphbittensor.png" width="1000" />
Above: Local loss function training in a k-graph shaped NN organization.

We follow this paradigmatic shift. Each connected computer within the network is training with respect to its own loss. Data can be continuously streamed through compute nodes, completely eliminating the wasted cycles spent blocking while waiting for error to return from some distant loss.

In this scheme, each node is constantly streaming message between it at its neighbors. They can immediately pull next examples from a queue during training and serve an inference model to adjacent downstream components, updating this model as they improve it over time.

The local models can be split width-wise in each node, across compute hardware with rapid communication, while the local losses allow depth-wise expansion, adding another dimension of parallelism to be exploited. More, the datasets are split as well, each node it responsible for its own corpus of language or images -- hypothetically increasing  model diversity.

## Market

_What is the product of Neuron_?

Abstractly, it must be the cell's ability to transform signal into actionable information within the mind.
This abstraction can be extended to a computing substrate as well -- the product of a machine intelligence unit, for instance, a Neural Network, is simply a mapping from an input to an output which converts unstructured signals into useful information.

But _useful_ to what? Intelligence is only a valuable commodity with respect to a problem. What problem should a global machine learning system work on?

We choose here unsupervised Language and Image modeling: where structured transformations in these domains -- ones that extract meaning or useful feature -- are used ubiquitously in a large variety of additional intelligence problems. Most human knowledge is stored in language and imagery, and there exists a near infinite quality of cheap unlabeled training data within both domains.

We gate access to this network using a digital token, allowing holders to maximize the performance on a downstream problem and paying contributing computers in the same token which holds this value.



## Organization

<img src="assets/brain_engineering_diagram.png" width="1000" />

Above: An Engineering diagram of the brain. For inspiration.

```

                                     [EOS]
                                       |
                                  [Metagraph]
                               /       |       \
                    ----------------------------------------
                  |                  Neuron                  |
                  |                                          |
                  | [Dendrite] ---> [Nucleus] ---> [Synapse] |
                  |                                          |
                  |                                          |
                    ----------------------------------------
                               \       |       /
                                     [Main]
```


###### Nucleus
The main Tensorflow graph is defined and trained within the Nucleus object. As is, the class is training a self supervised word-embedding over a dummy corpus of sentences in text8.zip. The result is a mapping which takes word to a 128 dimension vector, representing that word while maintaining its semantic properties.

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

<img src="assets/knowm.png" width="1000" />

The BitTensor network, in aggregate, forms a single meta Machine Learning model composed of a many smaller interconnected sub-graphs. The connections between these nodes reflect channels along which tensors are passed, containing, in the forward direction, features, and in the reverse direction gradients: No different than the individual layers of a standard Neural Network architecture (or Tensorflow graph)

<img src="assets/NN.jpeg" width="1000" />

Client nodes communicate upstream to servers and, while training their local objective function, produce attribution values, which are numerical evaluations of each connection. We use Fishers Information Metric to produce attributions in the standard code, but any method is sufficient. In total, aggregated attributions from the entire network describe a directed weighted graph (DWG) structure. (Below)

<img src="assets/weboftrust.jpg" width="1000" />

The DWG is updated discretely through emission transactions, and are conglomerated on the EOS blockchain. This process produces global attribution scores: between the full-network and each sub-grph. New tokens are distributed in proportion to this global-attributuon.

Below is a simulated version of that emission written in numpy:

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

## Word-Embeddings

A word-embedding is a projection from a word to a continuous vector space representation of that word, which attempts to maintain the semantics under the projection, For instance, 'King' --> [0,1, 0,9, ..., -1.2], such that 'King' - 'Queen' = 'Male'.

Word-embeddings are highly useful initital projections for a large number of Machine Learning problems. This makes them an ideal product for our network. They can also be trained in an un-supervised fashion which is a requirment for the local-loss approach described above.

During training we use a language corpus and find our projection by training a classifier to predict words in context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or ('hair' -> 'long'). The ability to predict context requires an understanding of the relationships between words ('meaning' is a relational quality) -- a highly successful assumption in practice.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding producing method is possible -- indeed, the goal should be diversity.

## License

MIT
