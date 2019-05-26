## Unsupervised Peer-to-Peer Machine Learning.

# About.
This is a research tool for working with unsuperivsed peer to peer Machine Learning systems. The goals is to find novel ways of combining computing 
resources which extend beyond a trusted computing boundary. The hope is that incentivization models like the ones in Bitcoin or BitTorrent can be used to
align the computing resources and keep them online and working towards a worthy Machine Learning goal. An unsupervised task seems ideal in this environemnt
because data is easy to come by and each computer can work on their own without the need for a supervised moderator. 

# What is the state of this project?

The server code spins up a Neuron object which is training a standard word embedding. This NN trains over a simple dataset in text8.zip to produce a 128 dim
word embedding which it exposed on a GRPC server. Currently, tracking and metagrah construction are not implemented, nor is an incentive structure which I 
hope will center around a easily transactable crypto currency. As is, you can specify another server on the local host and your node will combine the trained 
embedding from that of your own during training.

# To Run
```
$ git clone https://github.com/unconst/SimpleWord2Vec
$ cd SimpleWord2Vec
$ pip install -r requirements.txt
$ python server.py '[::]:50501' '[::]:50502'
$ python server.py '[::]:50502' '[::]:50501'
```

# About Word Embeddings.

A word embedding is a projection from a word to a continuous vector space 'cat' --> [0,1, 0,9, ..., -1.2], which attempts to maintain the word's semantics. 
For instance, 'King' - 'Queen' = 'Male'. Word embeddings are highly useful first order projections for a number of Machine Learning problems which makes 
them an ideal product for a p2p machine learning network attempting to be useful for the largest number of individuals. 

Word embeddings can be trained in a self-supervised manner on a language corpus by attempting to find a projection which helps a classifier predict word in 
it's context. For example, the sentence 'The queen had long hair', may produce a number of supervised training examples ('queen' -> 'had'), or 
('hair' -> 'long'). The assumption is that the meaning of a word is determined by the company it keeps. In practice this assumption has been highly succesful.

In the prototype node above, we train each NN using a standard skip gram model, to predict the following word from the previous, however any other embedding
producing method is possible. The goal is diversity.

```
def log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique,
                                  range_max, seed=None, name=None):
  """Samples a set of classes using a log-uniform (Zipfian) base distribution.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max)`.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution for this operation is an approximately log-uniform
  or Zipfian distribution:

  `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

  This sampler is useful when the target classes approximately follow such
  a distribution - for example, if the classes represent words in a lexicon
  sorted in decreasing order of frequency. If your classes are not ordered by
  decreasing frequency, do not use this op.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_candidate_sampling_ops.log_uniform_candidate_sampler(
      true_classes, num_true, num_sampled, unique, range_max, seed=seed1,
      seed2=seed2, name=name)

def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.

  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).

  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
        class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits_v2`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.
  Returns:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits_v2` (sampled softmax).
    out_labels: A Tensor object with the same shape as `out_logits`.
    """
```

