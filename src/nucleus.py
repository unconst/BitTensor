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

class Nucleus():
    def __init__(self, config, metagraph, dendrite):
        """ The main Tensorflow graph is defined and trained within the Nucleus object.
        As is, training a self-supervised word-embedding over
        a dummy corpus of sentences in text8.zip. The result is a mapping which
        takes a word to a 128 dimension vector; representing that word while
        maintaining its semantic properties.

        """
        # A Bittensor config object.
        self.config = config
        # A Metagraph object which maintains state about the bittensor network.
        self.metagraph = metagraph
        # Dendrite object responsible for GRPC connections to other nodes.
        self.dendrite = dendrite

        # A threading object which runs the training loop.
        self.main_thread = threading.Thread(target=self._main)
        self.main_thread.setDaemon(True)
        # A boolean set when the nucleus is training.
        self.running = False

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
        self.session = tf.compat.v1.Session(graph=self.graph)

        # Init Nucleus Graph and save model to data/$ID.
        self.model_checkpoint_dir = 'data/' + self.config.identity + '/model'

        # Init tables and vars.
        self.session.run(self.var_init)
        self.session.run(self.table_init)

        # Save the initial graph.
        self.saver.save(self.session, self.model_checkpoint_dir)
        logger.info('Saved initial inference graph to {}.', self.model_checkpoint_dir)


    def build_vocabulary(self):
        """ Parses the dummy corpus into a single sequential array.
        Each sentence is appended to each other. Also produces count dictionary
        for each word in the corpus.
        """
        # Read textfile.
        f = zipfile.ZipFile(self.filename)
        for name in f.namelist():
            self.words = tf.compat.as_str(f.read(name)).split()
        f.close()

        counts = [('UNK', -1)]
        counts.extend(collections.Counter(self.words).most_common(self.vocabulary_size - 2))
        self.string_map = [c[0] for c in counts]

        logger.debug('Built Nucleus vocabulary.')

    def build_graph(self):
        """ Builds the training and inference graphs.
        """

        # Global step.
        self.global_step = tf.compat.v1.train.create_global_step()

        # Boolean flag which determines whether or not we spike our downstream
        # nodes through the dendrite.
        # TODO(const) Add distillation networks for each dendrite.
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=[], name='is_training')

        #
        # ----
        #
        ### Below: Train Preproccessing

        # Input words.
        self.train_batch_words = tf.compat.v1.placeholder(tf.string, shape=[self.batch_size, 1], name="training_batch_words")
        self.train_batch_labels = tf.compat.v1.placeholder(tf.string, shape=[self.batch_size, 1], name="training_batch_labels")
        train_batch_words_rs = tf.reshape(self.train_batch_words, [self.batch_size])

        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.string_map), num_oov_buckets=1, default_value=0)
        word_ids = vocabulary_table.lookup(train_batch_words_rs)
        label_ids = vocabulary_table.lookup(self.train_batch_labels)

        # Get remote inputs. Blocking RPC which multicast queries upstream nodes.
        remote_inputs = self.dendrite.spike(self.is_training, tf.reshape(self.train_batch_words, [self.batch_size, 1]), self.embedding_size)
        remote_inputs = [tf.reshape(rmi, [self.batch_size, self.embedding_size]) for rmi in remote_inputs]

        queue_dtypes = [tf.int64] + [tf.int64] + [tf.float32] * self.config.k
        queue_shapes = [[self.batch_size, 1]] + [[self.batch_size]] + [[self.batch_size, self.embedding_size] for rimp in remote_inputs]
        self.dendrite_queue = tf.queue.FIFOQueue(
            capacity=100,
            dtypes=queue_dtypes,
            shapes=queue_shapes)

        # Pack the next example:
        pre_next_example = [label_ids] + [word_ids] + remote_inputs
        a = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        b = tf.zeros([self.batch_size], dtype=tf.int64)
        c = [tf.zeros([self.batch_size, self.embedding_size]) for _ in range(self.config.k)]
        #pre_next_example = [a] + [b] + c
        self.enqueue_op = self.dendrite_queue.enqueue(pre_next_example)


        ### Done: Train Preprocssing.
        #
        # ----
        #
        ### Below: Graph Inputs Switch.

        # Inference Inputs:
        # TODO(const) implement distillation pipeline here. During inference this should be run through the distilled network.
        self.inference_batch_words = tf.compat.v1.placeholder(tf.string, shape=[None, 1], name="inference_batch_words")
        inference_batch_words_rs = tf.reshape(self.inference_batch_words, [-1])
        inference_word_ids = vocabulary_table.lookup(inference_batch_words_rs)
        dummy_dendrite_inputs = self.dendrite.spike(self.is_training, tf.reshape(self.inference_batch_words, [-1, 1]), self.embedding_size)
        inference_inputs = [inference_word_ids] + [tf.zeros([1], dtype=tf.int64)] + dummy_dendrite_inputs

        # Switch between inference graph and training graph.
        next_inputs = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                    true_fn=lambda: self.dendrite_queue.dequeue(),
                    false_fn=lambda: inference_inputs)

        # batch_size should not be used passed this point.
        next_label_ids = tf.reshape(next_inputs[0], [-1, 1])
        next_word_ids = tf.reshape(next_inputs[1], [-1, 1])
        next_remote_inputs = [tf.reshape(rmi, [-1, self.embedding_size]) for rmi in next_inputs[2:]]

        ### Done: Graph Inputs Switch.
        #
        # ----
        #
        ### Below: Main Graph.

        # Embeddings Lookup.
        embeddings = tf.Variable(tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        word_embeddings = tf.nn.embedding_lookup(embeddings, next_word_ids)
        word_embeddings = tf.reshape(word_embeddings, [-1, self.embedding_size])

        # Full input layer.
        full_inputs = [word_embeddings] + next_remote_inputs
        l1 = tf.concat(full_inputs, axis=1)

        # Hidden Layer
        # TODO(const) More than one layer?
        w1 = tf.Variable(tf.random.uniform([self.embedding_size * (self.config.k + 1), self.embedding_size], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([self.embedding_size]))
        final_layer = tf.sigmoid(tf.matmul(l1, w1) + b1)

        # Embedding output.
        self.output = tf.identity(final_layer, name="embedding_output")

        # Embedding Weights
        softmax_weights = tf.Variable(tf.random.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Sampled Softmax Loss.
        batch_loss = tf.nn.sampled_softmax_loss(
            weights=softmax_weights,
            biases=softmax_biases,
            labels=next_label_ids,
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
            tf.compat.v1.summary.scalar('attribution' + str(i), input_attribution)

        # Average loss.
        self.loss = tf.reduce_mean(batch_loss)
        tf.compat.v1.summary.scalar('loss', self.loss)

        # Merge sumaries.
        self.merged_summaries = tf.compat.v1.summary.merge_all()

        # Convert PNG buffer to TF image
        self.metagraph_image_placeholder = tf.compat.v1.placeholder(dtype=tf.string)
        self.metagraph_image_buffer = tf.image.decode_png(self.metagraph_image_placeholder, channels=4)
        self.metagraph_summary = tf.compat.v1.summary.image("Metagraph State", tf.expand_dims(self.metagraph_image_buffer, 0))

        # Summary writer for tensorboard.
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.config.logdir, self.graph)

        # Optimizer.
        self.optimizer = tf.compat.v1.train.AdagradOptimizer(1.0).minimize(self.loss, global_step=self.global_step)

        # Init vars.
        self.var_init = tf.compat.v1.global_variables_initializer()
        self.table_init = tf.compat.v1.tables_initializer(name='init_all_tables')

        # Model Saver.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=2)

        logger.debug('Built Nucleus graph.')

    def get_feeds(self):
        """ Returns the feed dictionary for the preproccessing session thread.
        """
        # Build a random batch [feature = word_i, label = word_i+1]
        batch_words = []
        batch_labels = []
        for i in range(self.batch_size):
            index = random.randint(0, len(self.words) - 2)
            batch_words.append([self.words[index]])
            batch_labels.append([self.words[index + 1]])

        # Build Feeds dictionary.
        feeds = {
            self.train_batch_words: batch_words,
            self.train_batch_labels: batch_labels,
            self.inference_batch_words: [['UNK']], # Dummy input for placeholder.
            self.is_training: True
        }
        return feeds

    def get_fetches(self):
        """ Returns the model fetches for the training session thread.
        """
        # Build Fetches dictionary.
        fetches = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'summaries': self.merged_summaries,
            'attributions': self.attributions,
            'global_step': self.global_step
        }
        return fetches


    def _preprocess_loop(self):
        """ The _preprocess_loop is pulling batches from the vocabulary and
        passing them into the graph. This process is making the dendrite RPC
        calls into the network, the resulting inputs are appended to a FIFO
        queue which is being consumed by the _train_loop.
        """
        logger.info('Started preproccess thread.')
        try:
            with self.coord.stop_on_exception():
                while not self.coord.should_stop() and self.running:
                    run_output = self.session.run(  fetches=[self.enqueue_op],
                                                    feed_dict=self.get_feeds())
        except Exception as e:
            self.coord.request_stop(e)
        logger.info('Stopped _preprocess_loop thread.')


    def _train_loop(self):
        """ The _train_loop is pulling batches from the dendrite queue (preproccessing)
        these contain words, and the embeddings from upstread nodes int he network.
        The examples are used to train a local langauge model.

        The train loop also post summaries about training to the tensorboard
        server pertaining to training loss attribution scores.
        """
        # TODO(const): Attribution scores should be posted to tensorboard
        # through the validation thread, not through training.
        logger.info('Started training thread.')
        try:
            with self.coord.stop_on_exception():
                # Train loop.
                best_loss = math.inf
                while not self.coord.should_stop() and self.running:
                    run_output = self.session.run(  fetches=self.get_fetches(),
                                                    feed_dict=self.get_feeds())

                    # Unpack run_output.
                    self.train_step = run_output['global_step']
                    self.current_loss = run_output['loss']

                    # Set the attributions in the metagraph.
                    # TODO(const) Problematic! This is not thead safe.
                    self.metagraph.attributions = self.normalize_attributions(run_output['attributions'])

                    # Step iteration check. Only update vars every 200 steps.
                    if self.train_step % 200 == 0:
                        logger.debug('Step {}, Loss best:{}, Current:{}, Attrs {}',
                                    self.train_step,
                                    ("%.4f" % best_loss),
                                    ("%.4f" % self.current_loss),
                                    [(attr[0], "%.4f" % float(attr[1])) for attr in self.metagraph.attributions])

                        # Add Attribution summaries to tensorboard.
                        self.summary_writer.add_summary(run_output['summaries'], self.train_step)

                        # Add stake summaries to Tensorboard.
                        my_stake = self.metagraph.get_my_stake()
                        total_stake = self.metagraph.get_total_stake()
                        stake_fraction = float(my_stake) / float(total_stake)
                        my_stake_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="My Stake", simple_value=my_stake)])
                        total_stake_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Total Stake", simple_value=total_stake)])
                        stake_fraction_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Stake Fraction", simple_value=stake_fraction)])
                        self.summary_writer.add_summary(my_stake_summary, self.train_step)
                        self.summary_writer.add_summary(total_stake_summary, self.train_step)
                        self.summary_writer.add_summary(stake_fraction_summary, self.train_step)


                    # Log and save new inference graph if best.
                    if self.train_step % 500 == 0:
                        if run_output['loss'] < best_loss:
                            # Save new inference graph.
                            best_loss = run_output['loss']
                            model_checkpoint_dir = 'data/' + self.config.identity + '/model'
                            self.saver.save(self.session,
                                            model_checkpoint_dir,
                                            write_meta_graph=True)
                            logger.info('Saved new inference graph to {}.', self.model_checkpoint_dir)


        except Exception as e:
            self.coord.request_stop(e)
        logger.info('Stopped training thread.')


    def start(self):
        """ Begins the Nucleus main thread. This thread contains the coordinator
        and sub-threads for training, preprocessing, and validation.
        """
        self.running = True
        self.main_thread.start()

    def stop(self):
        """ Stops the Nucleus main thread. This recursively kills the training
        validation and preprocessing threads under the coordinator.
        """
        logger.debug('Request Nucleus stop.')
        self.running = False
        if self.coord:
            self.coord.request_stop()
        self.main_thread.join()

    def _main(self):
        """ The main Nucleus thread creates and contains a coordinator object
        with sub-threads to the training and pre-processing threads.
        """
        logger.debug('Started Nucleus training.')
        with self.session:
            try:
                # Set up threading coordinator.
                self.coord = tf.train.Coordinator()

                # TODO(const) Need to implement the validation loop here.

                # Create and start the training and preprocessing threads.
                preproccess_thread = threading.Thread(target=self._preprocess_loop)
                training_thread = threading.Thread(target=self._train_loop)
                preproccess_thread.setDaemon(True)
                training_thread.setDaemon(True)
                preproccess_thread.start()
                training_thread.start()

                # Wait on threads.
                self.coord.join([preproccess_thread, training_thread])

            except Exception as e:
                # Stop on exception.
                self.coord.request_stop(e)
                self.coord.join([preproccess_thread, training_thread])
                logger.error(e)

            finally:
                # Stop on all other exits.
                self.coord.request_stop()
                self.coord.join([preproccess_thread, training_thread])
                logger.debug('Stopped Nucleus training.')



    # (Bellow) Functions pretaining to the creation of a metagraph summary image.

    def update_metagraph_summary(self, metagraph_buffer):
        """ Updates the metagraph plot on tensorboard.
        metagraph_buffer is a image buffer of bytes created by calling
        visualization.generate_edge_weight_buffer(metagraph.nodes) in the
        main thread.
        """
        metagraph_summary = self.session.run(self.metagraph_summary, feed_dict={self.metagraph_image_placeholder: metagraph_buffer.getvalue()})
        self.summary_writer.add_summary(metagraph_summary, self.train_step)

    def normalize_attributions(self, attributions):
        """ Normalizes the attribution scores from the main graph, ensures that
        they sum to 1 and that they are only normalized over true inputs.
        """
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
