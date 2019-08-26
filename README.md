## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

> Decentralized Machine Intelligence

## Table of Contents

- [Overview](#overview)
- [To-Run-Locally](#to-run-locally)
- [To-Run-Testnet](#to-run-testnet)
- [Why](#why)
- [How](#how)
- [Market](#market)
- [Incentive](#incentive)
- [Organization](#organization)
  - [Nucleus](#nucleus)
  - [Dendrite](#dendrite)
  - [Synapse](#synapse)
  - [Metagraph](#metagraph)
- [Word-Embeddings](#word-embeddings)
- [License](#license)

---

## Overview

BitTensor allows a new class of Machine Learning model to train across a peer-to-peer network. It enables any computer and any engineer in the world to contribute in training.

The nature of trust-less computing necessitates that these contributions are combined through incentive rather than direct control from any one computer. We use a digital token to carry that incentive signal through the network: where the magnitude of this incentive is derived from a p2p collaborative filtering technique similar to Google's Page rank algorithm.  

The lack of centrality allows the structure to grow to arbitrary size across the internet. Both the cost and control of the system is distributed. And the network's informational product is priced into the reward token's value.

When run, this software folds your computing power into a p2p network and rewards you with an EOS based digital token for your contribution.

## To-Run-Locally
1. [Install Docker](https://docs.docker.com/install/)

```
$ git clone https://github.com/unconst/BitTensor
$ cd BitTensor

# Run a test EOS blockchain.
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

# Run a Remote Node
$ ./bittensor.sh --remote --token $DIGITAL_OCEAN_TOKEN --eosurl http://142.93.177.245:8888


# Run a Local Node
$ python src/upncp.py --port 9091  // To punch a hole in your router.
$ ./bittensor.sh --port 9091 --eosurl http://142.93.177.245:8888

```

<img src="assets/brain.png" width="1000" />

## Why

We believe Machine Intelligence, like Human Intelligence, is an a priori good. And yet, intelligence is power and power, if held in the hands of the few, will corrupt. It should be democratized and made open source. Unfortunately, companies like OpenAI who claim this goal have failed in their mandate, opening up the algorithms but not access to the _intelligence_ itself.

This technology is being built to do this, while democratizing its ownership, and sharing its profit with any computer or any individual who deems it worthwhile to contribute.

Moreover, although democratization and openness are ethical values, we are relying on their practical use here: A free and open system with a large number of stake holders is also the most direct path towards our goal of producing Strong Machine Intelligence. The scale of the AI problem in front of us necessitates that we build it this way.

Why is this? Because a decentralized computing approaches harness the largest pool of computing power and the largest pool of collaborators: Every computer and every engineer can contribute to this system.

We've seen this work with this technology's progenitors. Bitcoin is the largest super computer in the world, BitTorrent, at one time, the largest bandwidth user across the globe, and open source Linux, the most widely used operating system in use today.

<img src="assets/Lightning.png" width="1000" />

Above: Bitcoin Lightning network nodes from late 2018.

## Training

In standard Machine learning setting, the training mechanism uses Back-propagation to minimize the loss on a dataset with respect to the weights, and at each step the model parameters wait for a signal from the global loss function before the next update.

This loss function centrality is prohibitive when the scale of those networks reach the scale desired by modern machine learning  -- or biological scale -- and necessarily so, when we are attempting to train a system which spans multiple computers connected across the web, as we are doing here.

Instead, the network is trained using unique loss functions at each node. Each participant within the network is training against its own loss and against its own dataset. This allows us to train subsections of the network independently, dividing and conquering the problem so that each locality is not (immediately) dependent on far off events in the large network. 

<img src="assets/kgraphbittensor.png" width="1000" />
Above: Local loss function training in a k-graph shaped NN organization.

Training steps on each node run asyncronously. Each node serves an inference model to adjacent components, updating this model as they improve it over time. There is no need to wast cycles waiting for error to return from some distant loss. And each node is constantly streaming messages between it at its neighbors in the network. The local models can be split width-wise in each node, across compute hardware with rapid communication, while the local losses allow depth-wise expansion, adding another dimension of parallelism to be exploited. 

## Scoring

<img src="assets/knowm.png" width="1000" />

In order to prevent the need for the entire connected network to run at each step, each component is optimized locally – i.e performing a parameter update given only an input x and a target y drawn from its local dataset. It does not wait for a global error signal to propagate backwards from another computer.

In this setting each component contains a dataset with targets X and labels Y, and is attempting to ﬁt a function that predicts the output from the input, yˆ = f(x), by minimizing some loss metric on the output of the model, L(ˆy, y). 
Each component model is also a composition of its neighboring models f = (f 1 ◦ f 2 ...f n ), and so we are optimizing the parameters θ of this composition by moving them in the direction of the gradient of the loss, ∂ θ i L(ˆy, y). 

We can derive an approximation of importance for each neighboring component in the network. In this implementation we use an information-theoretic approach to determine the value of each composed function f 1 ◦ f 2 ...f n by using the Fisher
Information (FIM) as a proxy for importance. The Fisher criterion is a natural metric for quantifying the relative importance of inputs since it provides an estimate of how much information a set of parameters carry about the model's output distribution, namely, f(x).

FIM can be calculated using the covariance of the gradient of log likelihood with regards to the model's parameters θ. This can be calculated from the expectation of the element-wise multiplication of the gradient: FIM(θ) = Ey [g  g], where g = ∂ log f(y|x;θ) / ∂θ is the gradient of the log-likelihood and  represents element-wise multiplication. FIM(θ) is a p×1 vector, where p is the numebr of parameters, and each element is the Fisher Information of a that corresponding parameter.

The Fisher Information provides an estimate of the amount of information a random variable carries about a parameter
of the distribution. In the context of a compositional function, this provides a natural metric for quantifying the relative importance of a neighboring component in the network. The less information connected parameters to this input hold, the less important that component is to the output statistics of the network. A FIM score for each of our composed functions can derived by summing the parameter scores of all weights attached to that input.

## Ranking & Emmission

We rank individual components in the network based on the notion of transitive use: If a component i values a component j, it should also value the components trusted by j since component j is a composition of its neighbors. By using the method described above each component i calculates the local weight importance wij for all its neighbors. This is a reflection of how much the output of component i depends on the input from component j. 

These scores are normalized and posted to a centralized contract running on a decentralized append only database (blockchain).  This contract stores these normalized weights as a directed weighted graph G = [V, E] where for each edge _eij_ in E we have a value _wij_ associated with the connection between component _fi_ and _fj_. G is updated continuously by transaction calls from each working component as they train and as they calculate attribution scores for their neighbors in the network.

Using the local information in aggregate, we can derive a global attribution score for component _fi_, _ai_ which reflects its use to the entire network, rather than just its neigbors. A standard approach is the EigenTrust algorithm which iteratively updates the component use vector to an attractor point through multiple multiplications by the adjacency matrix described by G. i.e. a(t+1) = G * a(t). 

However this algorithm suffers standard attacks like whitewashing, where nodes continually join and leave the network, and sybil attacks where malicious users create many fake nodes to influence the ranking scores. We use a staking method to alleviate these concerns. In this setting nodes must attain access to a finite digital token and attach it to the address in use by the network node. It is not possible to create many wieght holding spurious nodes without access to this token. And, the conecept of identity is made permenant by hoding it fixed to a digital token account address. 

Global attribution scores derived using the method above give us a ranking for each node in the network. As a manner of incentivizing nodes to stay online we wish to incentivize them using a value holding token. These should be distributed first to computers which are producing value. 

An approximate method is written below using python-numpy:
```
def attribution_simulation():

    # Stake vector.
    S = [9.  8.  7.  6.  5.  4.  3.  2.  1.  0.]

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




---

## Word-Embeddings

A word-embedding is a projection from a word to a continuous vector space representation of that word, which attempts to maintain the semantics under the projection, For instance, 'King' --> [0,1, 0,9, ..., -1.2], such that 'King' - 'Queen' = 'Male'.

Word-embeddings are highly useful initital projections for a large number of Machine Learning problems. This makes them an ideal product for our network. They can also be trained in an un-supervised fashion which is a requirment for the local-loss approach described above.

During training we use a language corpus and find our projection by training a classifier to predict words in context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or ('hair' -> 'long'). The ability to predict context requires an understanding of the relationships between words ('meaning' is a relational quality) -- a highly successful assumption in practice.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding producing method is possible -- indeed, the goal should be diversity.

## License

MIT
