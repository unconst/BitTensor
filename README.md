## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

## Table of Contents

- [Overview](#overview)
- [To-Run-Locally](#to-run-locally)
- [To-Run-Testnet](#to-run-testnet)
- [Neurons](#neurons)
- [Method](#method)
  - [Representation](#representation)
  - [Incentive](#incentive)
  - [Attribution](#attribution)
  - [Emission](#emission)
  - [Market](#market)
- [References](#references)
- [License](#license)

---

## Overview

BitTensor allows a new class of Machine Learning model which trains across a peer-to-peer network. It enables any computer and any engineer in the world to contribute to its training.

The nature of trust-less computing necessitates that these contributions are driven by incentives rather than by direct control from any one computer. We use a digital token to carry that incentive signal through the network. The magnitude of this incentive is derived from a p2p collaborative filtering technique similar to Google's Page Rank algorithm.  

As a network product we focus on representation learning, starting from language and extending the network to image and speech. The result is a sufficiently general product which is useful to a large number of downstream stake holders.

The lack of centralization allows the structure to grow to an arbitrary size across the internet. Both the cost and control of the system is distributed and the network's informational product is priced into the reward token's value.

When run, this software integrates your computing power into a p2p network and rewards you with an EOS-based digital token for your contribution.

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
$ ./bittensor.sh --neuron boltzmann

...

# Run Node N.
$ ./bittensor.sh --neuron elmo

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

## Neurons

1. Boltzmann: Trainable Feed-forward NN, tokenizes requests on a word level basis responds with 128 dimensional embedding vectors. Applies gradients on 3 second interval without averages.

1. Feynmann: Trains a embedding model over the network using a dummy corpus from text.zip. Serves trained model to network. Does not apply gradients.

1. Elmo: Untrainable Elmo NN.

1. GoogleUSE: Google's universal sentence encoder. Non-trainable. Trained on question and answer text.


## Method
<img src="assets/brain.png" width="1000" />

### Introduction

A Neural Network easily decomposes into smaller sub-components where in the forward direction, each component receives activations from downstream, operates on them, and passes them forward. Then during training receives gradients from upstream and passes them backward.  A peer-to-peer version of a Neural Network is merely a composition of these sub-components except that message passing occurs across the Wide Area Network, no computer is privileged and there are no guarantees on the proper behavior of the composed network's constituent elements.

To describe this more formally, each sub-component across the network serves a function f(x) which is parameterized by θ, acts on inputs x, (of some type, text, image, etc) and produces outputs y = f(x) as tensors with some shape. Further more, the ith function may be a composition of downstream components, such that yi = fi(x, d0(x), d1(x), ..., dn(x)) and reflexively, the ith function can be one compositional component to a set of upstream components, u0, u1, ... uj ... uN = fj(x, ..., fi(x), ...)

We assume f(x) is differential and can be trained using the reverse accumulation of error "back-propagation". Here, the ith component accepts a gradient (x, dy), where dy = dyj/dyi, for the output yj of some upstream component ui, and produces gradients ddj = df(x)/ddj for each downstream neighbor dj.

<p align="center"> <img src="assets/UpDn.png" width="500" /> </p>

The tuple (x, dy), advises the ith component on how to update its parameters θ in order to minimize some loss term defined by an upstream peer. This is no different than the gradient calculations for individual components in any differential graph architecture built with TensorFlow or PyTorch. We use google's protocol buffers to carry messages across the web and GRPC to serve each function at the end points. Calls on this server are of two types Spike: forward queries or Grade: gradient queries. Each call is atomic and can be scaled laterally to run in parallel.

### Representation

For our p2p Neural Network architecture we require a Machine Intelligence problem which is general enough to interest a diverse set of stake holders. The problem should be sufficiently difficult to warrant a collaborative approach and the data used to train it should be ubiquitous and cheap.

For our purposes, we choose unsupervised representation learning [5, 6, 7, 9, 10, 14, 24, 25], where the network trains on large-scale unlabeled corpora to learn a feature basis ('representation') of inputs. These representations are a form of product, which uniquely identifies and disentangles the underlying explanatory factors of an input (a.k.a an inductive bias). This is a widely used product and arguably a fundamental task in the development of an AI which understands the world around it. [26]

We initially focus on Language Representation from text, where components build an understanding of natural language and will respond to queries in pure unicode strings with a vector representation. For the sake of generality, we leave tokenization and parsing to each component and limit outputs across the network to fixed length vectors.

<p align="center"> "raw natural language text"  ---> [ f(T) ]  --->  [fixed length representation] </p>

The standard scheme for learning a representation is as follows. First, the raw text is tokenized, for instance at the word[32], sentence[31], or byte level[7], to create a sequence of discrete tokens T = (t1, t2, . . . , tn ). The modeling task consists in learning a representation for that sequence f(T) = f(t1, t2, . . . , tn).

Our representation function can be trained in any number of supervised or unsupervised forms. However, commonly this is achieved in an unsupervised manner, where a Neural Network architecture parameterized by θ, and trains f(T) to help predict other tokens T' in its near context.

<p align="center">  maximize ∏ P (T' | f(T)) </p>

This high-level paradigm has been shared and successfully applied by a large number of models to build incredibly powerful representations for language. These include:

<ul>
<li> ELMO [31] representations from bidirectional recurrent neural networks.

<li>BERT [5] representations from multi-word masking strategies.</li>

<li>MT-DNN [14] representations from multi-task knowledge transfer.</li>

<li>GPT-2 [7] representations from a question-answering dataset.</li>

<li>XLM representations from cross-lingual tasks.</li>

<li>ERNIE [10] representations entity/phrase level masking. </li>

<li>XLNet [9] representations across all mask permutations.</li>

</ul>

As an aside, the idea of learning representations in context, by attempting to predict the future or the past, was nicely described by John Rupert Firth in 1957 with the following quip: "you shall know a word by the company it keeps". It could be argued that the human mind is constantly attempting to do the same, namely, constructing a representation of the world which allows us to predict the future from the present -- a useful task for any organism.

### Incentives

We are extending previous work in Neural Network training by moving the training process from a datacenter into a decentralized computing domain where no computer is privileged, there is no single user of the network, and some computers may be incompetent, offline, or malicious. Because of these constraints, we must use _incentive_ to draw our compute nodes into line. That incentive should drive them to stay online, to learn well, and to train in alignment with a useful network product.

To begin, we assume a standard training scheme for the ith p2p component. It contains a dataset M, with targets X and labels Y, and is attempting to ﬁt a function that predicts the output from the input, yˆ = f(x), by minimizing the loss on the output of the model,

  <p align="center"> ith loss = Li = Ep[ Q(f(x), x) ]. (1) </p>

Where Q is a loss calculation between the targets and outputs and Ep is the expectation over a subset P of our full dataset M. Then a global objective for the entire network, G, should be to minimize the sum over each local objective.

Furthermore, it makes sense to scale our global objective with a stake vector S. This binds the concept of value into the network training process -- attaching more stake towards a loss function directly changes the global objective. Then for Si, the stake attached to the ith loss, we have:

  <p align="center"> global objective = G = min Σ Si ◦ Li . (2) </p>

### Attribution
We wish to mint new tokens to components in-proportion to their contribution to optimizing the _Global Objective_. We do this by asking what it would cost, in terms of loss, to prune each node from the network.

<p align="center"> ∆Lj = the change in global objective w.r.t removal of single component j. </p>

An approximation of ∆Lj can be attained by working first with each local estimation, ∆Lij, between a single loss Li, and a connected component j, and then transitively deriving all pair-wise paths through the network. We can calculate ∆Lij using a 2nd order approximation of the loss with respect to its input activations aj, and a change ∆aj reflecting the removal of the component j.

<p align="center"> ∆Lij = L(aj + ∆aj) − L(aj) ≈ g' ∙ ∆aj  +  1/2 ∆aj ∙ H ∙ ∆aj (4) </p>

Where g is the gradient of the loss which vanishes if we assume the loss is at a local optimum and the remaining term is the Hessian which can be approximated using an expectation over our training subset P:

<p align="center"> H ≈ Ep [ ( ∂L(x)/∂aj ) ^2] (6)</p>

This approximation becomes exact when P and M are close and Eqn. (6) can be viewed as an empirical estimate of the Fisher information of our activations. [17] We can use N data points to estimate our pruning signal *∆Lij*.

<p align="center"> gn = ( ∂L(xn)/∂aj )^2. </p>
<p align="center"> ∆Lij = 1/2N   ∆aj   Σn gn^2 (7)</p>

This information is available during the backward pass of computing the network’s gradient and the pruning signal can therefore be found at little extra computational cost.

Note, various other forms of pruning signals can be used, for instance [27][29][30]

### Emission

The totality of ∆Lij scores describe a directed weighted graph G = [V, E] where for each edge eij in E we have a the weight *∆Lij* associated with the connection between component i and j. ∆Lij is a local attribution and we would like to determine the global attribution for a node i, ∆Li. This score should be a sum over every pair-wise path through the graph weighted by stake *Si*.

<p align="center"> ∆Lj = Σi Si x ∆Lij </p>

We can derive all pair-wise paths by applying the chain rule to (7) to find the following transitive relation:

<p align="center"> Given ∆Lij and ∆Ljk </p>
<p align="center"> ∆Lik = ∆Lij x ∆Ljk (8) </p>

Which is intuitive, following immediately from the notion of transitive contribution: If a component i contributes to component j, it should multiplicatively contribute to the components using j since they are compositions of its parent. This transitive quality is also exploited in [29] for computing neural importance weights.

As a corollary of (8) global attribution scores for component _i_ can be calculated with a Power Iteration over the adjacency matrix described by G.

<p align="center">  ∆L(t+1) = G ∙ ∆L (t). (9) </p>

This is similar to the EigenTrust algorithm [23] or Google Page Rank [1], but with Fishers Information scores instead of recommendations or web links. We emit new tokens within the graph to components with high contribution scores proportionally. The entire calculation is done using a consensus engine which ensures that the specifics of token emission stay fixed and where the state of G is held global so that every node can see how they are attaining token emissions.

We note that this form of local attribution emission scheme may be similar in nature to the propagation of the neurotrophic factor BDNF in the brain.[2]

Below is an approximate method written using python-numpy:
```
N_BLOCKS = 100
DEPTH = 100
TOKENS_PER_BLOCK = 50

def get∆L ():
  ∆L = np.multiply (S, Lii)
  T = np.matmul (Lij, S)
  for _ in range(DEPTH):
      ∆L += np.multiply(T, Lii)
      T = np.matmul (Lij, T)

  ∆L = ∆L / np.linalg.norm(∆L, 1)
  return ∆L

def emit():
  # Stake vector.
  S = [9.  8.  7.  6.  5.  4.  3.  2.  1.  0.]

  # i to i attributions
  Lii = [0.6 0.9 0.4 0.5 0.5 0.5  1.  1.  1.  1. ]

  # i to j attributions
  Lij = [[0.  0.1 0.3 0.2 0.  0.  0.  0.  0.  0. ]
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
  for _ in range(N_BLOCKS):        
    ∆L = get∆L()                          // Modified EigenTrust
    E = ∆L * TOKENS_PER_BLOCK             // Emission this block
    S = S + E                             // Stake update.
```


### Market

A finite quantity of stake, sj is being allocated to each loss term in the network. This is global knowledge which when combined with local ∆Lij calculations and the deterministic emission system described above, produces a stream of newly minted tokens for each component in the network. The magnitude of the stream from upstream jth component to the downstream ith is wji.

The ith component is making discrete SGD steps to its parameters θ. These updates are made w.r.t gradient information passed from upstream components. We assume SGD steps are being made in proportion to individual stream scores wji such that the next parameter update is a scaled summation of the upstream gradients ∆θ1, ∆θ2 ... ∆θN:

<p align="center"> ∆θ = ∆θ1 (w1i/Σjwji) + ∆θ2 (w2i/Σjwji) + ... + ∆θN (wNi/Σj wji) (10) </p>  

For simplicity we imagine this as a single update over a single example, but in reality these updates are over batches of different size and w.r.t different text inputs. This construction is a proportional allocation game. Where from (4) we know the utility for the jth component:

<p align="center"> ∆Lj= L(θ + ∆θ) − L(θ) ≈ g' ∙ ∆θ  +  1/2 ∆θ ∙ H ∙ ∆θ (11) </p>

and ∆Lj is strictly increasing and continuous in wji. We have by [--] that there exists a unique optimal solution to bids wij where the sum of weighted losses (global loss) is minimized. Omitting the utility function overlap i.e. gradient steps may harm or help each other. We know from [--] that the market is at least 75% percent efficient. And if we introduce a Vicker Clark Groves (VCG) pricing method then the Nash Equilibrium is optimal. With VCG pricing the optimal bidding for each component is its raw utility, or exactly ∆Lij.

----


## References

[1] The PageRank Citation Ranking <br/>
http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf

[2] Brain-derived neurotrophic factor and its clinical implications <br/>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4697050/

[3] Attention is all you need <br/>
https://arxiv.org/abs/1706.03762

[4] Universal Language Model Fine-Tuning for Text Classification <br/>
https://arxiv.org/abs/1801.06146

[5] Bi-directional Encoder Representations from Transformers <br/>
https://arxiv.org/abs/1810.04805

[6] Googles Transformer-XL <br/>
https://arxiv.org/abs/1901.02860

[7] Open AI GPT2 <br/>
https://openai.com/blog/better-language-models/

[9] XLNet <br/>
https://arxiv.org/abs/1906.08237

[10] ERNIE: Enhanced Representation through Knowledge Integration <br/>
https://arxiv.org/abs/1904.09223

[11] RoBerta: A Robustly Optimized Bert Pre-training Approach <br/>
https://arxiv.org/abs/1907.11692

[12] Cross-lingual Language Model Pre-training <br/>
https://arxiv.org/pdf/1901.07291.pdf

[13] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer<br/>
https://arxiv.org/abs/1701.06538

[14] One Model To Learn Them All<br/>
https://arxiv.org/abs/1706.05137

[15] AHaH Computing–From Metastable Switches to Attractors to Machine Learning<br/>
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085175

[16] Distilling the Knowledge in a Neural Network <br/>
https://www.cs.toronto.edu/~hinton/absps/distillation.pdf

[17] Faster Gaze Prediction With Dense Networks and Fisher Pruning <br/>
https://arxiv.org/pdf/1801.05787.pdf

[18] Overcoming catastrophic forgetting in neural networks <br/>
https://arxiv.org/abs/1612.00796

[19] Bitcoin: A Peer-to-Peer Electronic Cash System <br/>
https://bitcoin.org/bitcoin.pdf

[20] IPFS - Content Addressed, Versioned, P2P File System <br/>
https://arxiv.org/abs/1407.3561

[21] Self-Attention with Relative Position Representations <br/>
https://arxiv.org/pdf/1803.02155.pdf

[22] Generating Wikipedia by Summarizing long sequences <br/>
https://arxiv.org/pdf/1801.10198.pdf

[23] The EigenTrust Algorithm for Reputation Management in P2P Networks <br/>
http://ilpubs.stanford.edu:8090/562/1/2002-56.pdf

[24] Distributed Representations of Words and Phrases and their Compositionality <br/>
https://arxiv.org/pdf/1310.4546.pdf

[25] Skip-Thought Vectors <br/>
https://arxiv.org/abs/1506.06726

[26] Representation Learning: A Review and New Perspectives <br/>
https://arxiv.org/pdf/1206.5538.pdf

[27] Optimal Brain Damage  <br/>
http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf

[28] A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks <br/>
https://arxiv.org/abs/1811.06031

[29] NISP: Pruning Networks using Neuron Importance Score Propagation <br/>
https://arxiv.org/pdf/1711.05908.pdf

[30] Overcoming catastrophic forgetting in neural networks <br/>
https://arxiv.org/abs/1612.00796

[31] Deep contextualized word representations <br/>
https://arxiv.org/pdf/1802.05365.pdf

[32] Efficient Estimation of Word Representations in Vector Space <br/>
https://arxiv.org/abs/1301.3781

## License

MIT
