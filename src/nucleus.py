import collections
import copy
from loguru import logger
import math
import numpy as np
import random
import tensorflow as tf
import time
import threading
import zipfile

import visualization


_ONE_DAY_IN_SECONDS = 60*60*24

class Nucleus():
    def __init__(self, config, metagraph, dendrite):
        """ The main Tensorflow graph is defined and trained within the Nucleus object.
        As is, the class is training a self supervised word embedding over
        a dummy corpus of sentences in text8.zip. The result is a mapping which
        takes word to a 128 dimension vector, representing that word while
        maintaining its semantic properties.

        """
        # A Bittensor config object.
        self.config = config
        # A Metagraph object which maintains state about the bittensor network.
        self.metagraph = metagraph
        # Dendrite object responsible for GRPC connections to other nodes.
        self.dendrite = dendrite

        # A threading object which runs the training loop.
        self.train_thread = threading.Thread(target=self._train)
        self.train_thread.setDaemon(True)
        # A boolean set when the nucleus is training.
        self.running = False
        # A mutex object for protecting concurrent calls to the nucleus objects.
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

        # Global step.
        self.global_step = tf.train.create_global_step()

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
        for i in range(self.config.k + 1):
            input_i = full_inputs[i]
            input_attribution = tf.abs(tf.reduce_sum(tf.gradients(xs=[input_i], ys=self.output)))
            self.attributions.append(input_attribution)
            tf.summary.scalar('attribution' + str(i), input_attribution)

        # Average loss.
        self.loss = tf.reduce_mean(batch_loss)
        tf.summary.scalar('loss', self.loss)

        # Merge sumaries.
        self.merged_summaries = tf.summary.merge_all()

        # Convert PNG buffer to TF image
        self.metagraph_image_placeholder = tf.placeholder(dtype=tf.string)
        self.metagraph_image_buffer = tf.image.decode_png(self.metagraph_image_placeholder, channels=4)
        self.metagraph_summary = tf.summary.image("Metagraph State", tf.expand_dims(self.metagraph_image_buffer, 0))

        self.summary_writer = tf.summary.FileWriter(self.config.logdir, self.graph)

        # Optimizer.
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss, global_step=self.global_step)

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

    def get_feeds(self):
        # Build a random batch [feature = word_i, label = word_i+1]
        batch_words = []
        batch_labels = []
        for i in range(self.batch_size):
            index = random.randint(0, len(self.words) - 2)
            batch_words.append([self.words[index]])
            batch_labels.append([self.words[index + 1]])

        # Build Feeds dictionary.
        feeds = {
            self.batch_words: batch_words,
            self.batch_labels: batch_labels,
            self.is_training: True
        }
        return feeds

    def get_fetches(self):
        # Build Fetches dictionary.
        fetches = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'summaries': self.merged_summaries,
            'attributions': self.attributions,
            'global_step': self.global_step
        }
        return fetches

    def update_metagraph_summary(self, metagraph_buffer):
        metagraph_summary = self.session.run(self.metagraph_summary, feed_dict={self.metagraph_image_placeholder: metagraph_buffer.getvalue()})
        self.summary_writer.add_summary(metagraph_summary, self.train_step)

    def normalize_attributions(self, attributions):

        attr_sum = attributions[0]
        for i in range(len(self.dendrite.channels)):
            node_i = self.dendrite.channels[i]
            if node_i:
                attr_sum += attributions[i+1]

        # Normalize attributions across non null edges.
        edges = [self.config.identity]
        norm_attributions = [attributions[0]/attr_sum]
        for i in range(len(self.dendrite.channel_nodes)):
            node_i = self.dendrite.channel_nodes[i]
            attr_i = attributions[i+1]/attr_sum
            if node_i:
                edges.append(node_i.identity)
                norm_attributions.append(attr_i)
        return list(zip(edges, norm_attributions))

    def _train(self):
        logger.debug('Started Nucleus training.')

        with self.session:

            # Init tables and vars.
            self.var_init.run()
            self.table_init.run()

            # Save the initial graph.
            self.saver.save(self.session, 'data/' + self.config.identity + '/model')

            # Train loop.
            self.train_step = 0
            best_loss = math.inf
            prev_image_buff = None
            while self.running:
                run_output = self.session.run(  fetches=self.get_fetches(),
                                                feed_dict=self.get_feeds())

                self.train_step = run_output['global_step']
                self.current_loss = run_output['loss']
                self.metagraph.attributions = self.normalize_attributions(run_output['attributions'])

                # Step iteration check. Only update vars every 200 steps.
                if self.train_step % 50 == 0:
                    logger.debug('Loss at step {}: {} -- attributions {}',
                                self.train_step,
                                self.current_loss,
                                self.metagraph.attributions)

                    # Add Attribution summaries to tensorboard.
                    self.summary_writer.add_summary(run_output['summaries'], self.train_step)

                    # Add stake summaries to Tensorboard.
                    my_stake = self.metagraph.get_my_stake()
                    total_stake = self.metagraph.get_total_stake()
                    stake_fraction = float(my_stake) / float(total_stake)
                    my_stake_summary = tf.Summary(value=[tf.Summary.Value(tag="My Stake", simple_value=my_stake)])
                    total_stake_summary = tf.Summary(value=[tf.Summary.Value(tag="Total Stake", simple_value=total_stake)])
                    stake_fraction_summary = tf.Summary(value=[tf.Summary.Value(tag="Stake Fraction", simple_value=stake_fraction)])
                    self.summary_writer.add_summary(my_stake_summary, self.train_step)
                    self.summary_writer.add_summary(total_stake_summary, self.train_step)
                    self.summary_writer.add_summary(stake_fraction_summary, self.train_step)


                # Log and save new inference graph if best.
                if run_output['loss'] < best_loss:
                    # Save new inference graph.
                    best_loss = run_output['loss']
                    self.saver.save(self.session,
                                    'data/' + self.config.identity + '/model',
                                    write_meta_graph=True)



        logger.debug('Stopped Nucleus training.')
