import collections
from loguru import logger
import math
from matplotlib import pylab
import numpy as np
import random
import tensorflow as tf
import time
import threading
import zipfile


_ONE_DAY_IN_SECONDS = 60*60*24

class Nucleus():
    def __init__(self, config, metagraph, dendrite):
        self.config = config
        self.metagraph = metagraph
        self.dendrite = dendrite

        self.train_thread = threading.Thread(target=self._train)
        self.train_thread.setDaemon(True)
        self.running = False
        self.mutex = threading.Lock()

        # Dataset zip file.
        self.filename = 'text8.zip'
        # Size of vocab embedding.
        self.vocabulary_size = 50000
        # Size of training batch.
        self.batch_size = 128
        # Dimension of the embedding vector.
        self.embedding_size = 128
        # Number of negative examples to sample.
        self.num_sampled = 64

        # Build Dataset.
        self.build_vocabulary()

        # Build Graph.
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.build_graph()

        # Create TF Session.
        self.session = tf.Session(graph=self.graph)

    def build_vocabulary(self):

        # Read textfile.
        f = zipfile.ZipFile(self.filename)
        for name in f.namelist():
            self.words = tf.compat.as_str(f.read(name)).split()
        f.close()

        counts = [('UNK', -1)]
        counts.extend(collections.Counter(self.words).most_common(self.vocabulary_size - 2))
        self.string_map = [c[0] for c in counts]

        #print (self.string_map)
        logger.debug('built neuron vocabulary.')

    def build_graph(self):

        # Boolean flag which determines whether or not we spike our downstream
        # nodes through the dendrite.
        # TODO(const) Add distillation networks for each dendrite.
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # Input words.
        self.batch_words = tf.placeholder(tf.string, shape=[None, 1], name="batch_words")
        self.batch_labels = tf.placeholder(tf.string, shape=[None, 1], name="batch_labels")
        batch_words_rs = tf.reshape(self.batch_words, [-1])

        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.string_map), num_oov_buckets=1, default_value=0)
        word_ids = vocabulary_table.lookup(batch_words_rs)
        label_ids = vocabulary_table.lookup(self.batch_labels)

        # Embeddings Lookup.
        embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        word_embeddings = tf.nn.embedding_lookup(embeddings, word_ids)

        # Get remote inputs. Blocking RPC which multicast queries upstream nodes.
        remote_inputs = self.dendrite.spike(self.is_training, tf.reshape(self.batch_words, [-1, 1]), self.embedding_size)

        # Full input layer.
        full_inputs = [word_embeddings] + remote_inputs
        l1 = tf.concat(full_inputs, axis=1)

        # Hidden Layer
        w1 = tf.Variable(tf.random_uniform([self.embedding_size * (self.config.k + 1), self.embedding_size], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([self.embedding_size]))

        final_layer = tf.sigmoid(tf.matmul(l1, w1) + b1)

        # Embedding output.
        self.output = tf.identity(final_layer, name="embedding_output")

        # Embedding Weights
        softmax_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Sampled Softmax Loss.
        batch_loss = tf.nn.sampled_softmax_loss(
            weights=softmax_weights,
            biases=softmax_biases,
            labels=label_ids,
            inputs=self.output,
            num_sampled=self.num_sampled,
            num_classes=self.vocabulary_size,
            num_true=1,
            sampled_values=None,
            remove_accidental_hits=True,
            partition_strategy='mod',
            name='sampled_softmax_loss',
            seed=None)

        # FIM (attribution) calculations
        self.attributions = []
        self.attribution_ops = []
        ema = tf.train.ExponentialMovingAverage(decay=0.98)
        for i in range(self.config.k + 1):
            input_i = full_inputs[i]
            input_attribution = tf.abs(tf.reduce_sum(tf.gradients(xs=[input_i], ys=self.output)))
            self.attribution_ops.append(ema.apply([input_attribution]))
            self.attributions.append(ema.average(input_attribution))

        # Average loss.
        self.loss = tf.reduce_mean(batch_loss)

        # Optimizer.
        with tf.control_dependencies(self.attribution_ops):
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

        # Init vars.
        self.var_init = tf.global_variables_initializer()
        self.table_init = tf.tables_initializer(name='init_all_tables')

        # Model Saver.
        self.saver = tf.train.Saver(max_to_keep=2)

        logger.debug('built neuron graph.')

    def start(self):
        self.running = True
        self.train_thread.start()

    def stop(self):
        self.running = False
        self.train_thread.join()

    def _train(self):
        logger.debug('started neuron training.')

        with self.session:

            # Init tables and vars.
            self.var_init.run()
            self.table_init.run()

            # Save the initial graph.
            self.saver.save(self.session, 'data/' + self.config.identity + '/model')

            # Train loop.
            average_loss = 0
            best_loss = math.inf
            step = -1
            while self.running:
                step += 1

                # Build a random batch [feature = word_i, label = word_i+1]
                batch_words = []
                batch_labels = []
                for i in range(self.batch_size):
                    index = random.randint(0, len(self.words) - 2)
                    batch_words.append([self.words[index]])
                    batch_labels.append([self.words[index + 1]])

                # Train Step.
                feed_dict = {self.batch_words: batch_words, self.batch_labels: batch_labels, self.is_training: True}
                out = self.session.run([self.optimizer, self.loss] + self.attributions, feed_dict=feed_dict)
                average_loss += out[1]

                # Progress notification and model update.
                if step % 200 == 1 and step > 200:
                    if average_loss < best_loss:
                        best_loss = average_loss
                        self.saver.save(self.session, 'data/' + self.config.identity  + '/model', write_meta_graph=True)

                    # TODO(const) this is sloppy. Should be called on a timed thread.
                    eval_attributions = []
                    for val in out[2:]:
                        eval_attributions.append(val)

                    self.metagraph.set_attributions(self.dendrite.channel_nodes, eval_attributions)

                    logger.debug('average loss at step {}: {} -- attributions {}', step, average_loss/200, eval_attributions)
                    average_loss = 0

        logger.debug('stopped neuron training.')
