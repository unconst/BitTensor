import collections
from loguru import logger
import random
import tensorflow as tf
import zipfile


class Nucleus():

    def __init__(self, hparams):
        self._hparams = hparams
        self._build_vocabulary()
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())
            self._session.run(tf.compat.v1.tables_initializer(
                name='init_all_tables'))

    def next_batch(self, batch_size):
        batch_words = []
        batch_labels = []
        for i in range(batch_size):
            index = random.randint(0, len(self._words) - 2)
            batch_words.append([self._words[index]])
            batch_labels.append([self._words[index + 1]])
        return batch_words, batch_labels

    def gate_outputs(self, spikes):
        # Run gating to get outgoing tensors.
        gate_feeds = {self._spikes: spikes}
        gate_outputs = self._session.run(self._gate_outputs, gate_feeds)
        return gate_outputs

    def run_synthetic_graph(self, spikes, targets, apply_step):
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: True,
        }

        # Build fetches.
        fetches = {
            'target_loss': self._target_loss, # Target accuracy.
            "cgrads" : self._cgrads, # child gradients.
        }
        if apply_step:
            fetches['target_step'] = self._tstep

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Return the batch accuracy.
        return run_output

    def run_graph(self, spikes, targets, cspikes, apply_step):
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: False
        }
        for i, cspike in enumerate(cspikes):
            feeds[self._cspikes[i]] = cspikes[i]

        # Build fetches.
        fetches = {
            'target_loss': self._target_loss, # Target accuracy.
            'load': self._load, # channel load.
            'synthetic_loss': self._syn_loss, # synthetic model loss.
            'child_grads': self._cgrads, # child gradients.
        }
        if apply_step:
            fetches['local_step'] = self._local_step
            fetches['gate_step'] = self._gate_step
            fetches['synthetic_step'] = self._synthetic_step

        # Return fetches.
        return self._session.run(fetches, feeds)


    def train(self, spikes, dspikes, targets):
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: False
        }
        for i in range(self._hparams.n_children):
            feeds[self._dspikes[i]] = dspikes[i]

        fetches = {
            "downstream_grads" : self._tdgrads,
            "target_loss" : self._target_loss,
            "train_step" : self._tstep,
            "synthetic_step" : self._syn_step,
            "scores" : self._scores
        }
        run_output = self._session.run(fetches, feeds)

        return run_output['downstream_grads'], run_output['target_loss'], run_output['scores']

    def spike(self, uspikes, dspikes, use_synthetic):

        # Build Feeds dictionary.
        feeds = {self._spikes : uspikes}
        feeds[self._use_synthetic] = use_synthetic
        for i in range(self._hparams.n_children):
            feeds[self._dspikes[i]] = dspikes[i]

        fetches = {'embedding' : self._embedding}
        if use_synthetic:
            fetches['synthetic_step'] = self._syn_step

        return self._session.run(fetches, feeds)['embedding']

    def grade(self, ugrades, uspikes, dspikes):

        # Build Feeds dictionary.
        feeds = {}
        feeds[self._spikes] = uspikes
        feeds[self._egrads] = ugrades
        for i in range(self._hparams.n_children):
            feeds[self._dspikes[i]] = dspikes[i]

        # Compute gradients for the children and apply the local step.
        dgrads = self._session.run([self._dgrads, self._estep], feeds)[0]

        # Return downstream grads
        return dgrads

    def _model_fn(self):

        # Placeholders.
        # Spikes: inputs from the dataset of arbitrary batch_size.
        self._spikes = tf.compat.v1.placeholder(tf.string,
                                                [None, 1],
                                                's')

        # Dspikes: inputs from previous component. Size is the same as the embeddings produced
        # by this component.
        self._dspikes = []
        for _ in range(self._hparams.n_children):
            self._dspikes.append(
                tf.compat.v1.placeholder(tf.float32,
                                         [None, self._hparams.n_embedding],
                                         'd'))
        # Egrads: Gradient for this components embedding, passed by a parent.
        self._egrads = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_embedding], 'g')
        # Targets: Supervised signals used during training and testing.
        self._targets = tf.compat.v1.placeholder(
            tf.string, [None, self._hparams.n_targets], 't')
        # use_synthetic: Flag, use synthetic downstream spikes.
        self._use_synthetic = tf.compat.v1.placeholder(tf.bool,
                                                       shape=[],
                                                       name='use_synthetic')

        # Joiner weights and biases.
        jn_weights = {
            'jn_w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_embedding * self._hparams.n_children,
                        self._hparams.n_jhidden1
                    ],
                                               stddev=0.01)),
            'jn_w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_jhidden1, self._hparams.n_jhidden2],
                        stddev=0.01)),
            'jn_w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_jhidden2, self._hparams.n_embedding],
                        stddev=0.01)),
        }
        jn_biases = {
            'jn_b1':
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_jhidden1])),
            'jn_b2':
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_jhidden2])),
            'jn_b3':
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_embedding])),
        }
        jn_vars = list(jn_weights.values()) + list(jn_biases.values())

        # Synthetic weights and biases.
        syn_weights = {
            'syn_w1':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_embedding, self._hparams.n_shidden1],
                        stddev=0.1)),
            'syn_w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_shidden1, self._hparams.n_shidden2],
                        stddev=0.1)),
            'syn_w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_shidden2, self._hparams.n_embedding],
                        stddev=0.1)),
        }
        syn_biases = {
            'syn_b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden1])),
            'syn_b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden2])),
            'syn_b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding])),
        }
        syn_vars = list(syn_weights.values()) + list(syn_biases.values())

        # Model weights and biases
        l_weights = {
            'w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_embedding + self._hparams.n_embedding,
                        self._hparams.n_hidden1
                    ],
                                               stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden1, self._hparams.n_hidden2],
                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden2, self._hparams.n_embedding],
                        stddev=0.1)),
            'w4':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_embedding, self._hparams.n_targets],
                        stddev=0.1)),
        }
        l_biases = {
            'b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding])),
            'b4':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        l_vars = list(l_weights.values()) + list(l_biases.values())


        # Target weights and biases.
        t_weights = {
            't_w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_vocabulary,
                        self._hparams.n_embedding
                    ]))
        }
        t_biases = {
            't_b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_vocabulary])),
        }
        ltvars = list(t_weights.values()) + list(t_biases.values())


        # Tokenizer weights and biases.

        # Tokenization with loopup table. This is the simplest form of
        # tokenization which simply looks up the word in a table to retrieve a
        # 1 x vocabulary sized vector.
        # string map, is a list of strings ordered by count.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(self._string_map),
            num_oov_buckets=1,
            default_value=0)

        # Token embedding matrix is a matrix of vectors. During lookup we pull
        # the vector corresponding to the 1-hot encoded vector from the
        # vocabulary table.
        embedding_matrix = tf.Variable(
            tf.random.uniform([self._hparams.n_vocabulary, self._hparams.n_embedding], -1.0,
                              1.0))


        # Tokenizer network.
        text = tf.reshape(self._spikes, [-1])
        labels = tf.reshape(self._targets, [-1])

        # Apply tokenizer lookup.
        text_tokens = vocabulary_table.lookup(text)
        label_tokens = vocabulary_table.lookup(labels)

        # Apply table lookup to retrieve the embedding.
        text_embedding = tf.nn.embedding_lookup(embedding_matrix, text_tokens)
        text_embedding = tf.reshape(text_embedding, [-1, self._hparams.n_embedding])

        # Joiner network.
        if self._hparams.use_joiner_network:
            dspikes_concat = tf.concat(self._dspikes, axis=1)
            jn_hidden1 = tf.nn.relu(
                tf.add(tf.matmul(dspikes_concat, jn_weights['jn_w1']),
                       jn_biases['jn_b1']))
            jn_hidden2 = tf.nn.relu(
                tf.add(tf.matmul(jn_hidden1, jn_weights['jn_w2']),
                       jn_biases['jn_b2']))
            jn_embedding = tf.add(tf.matmul(jn_hidden2, jn_weights['jn_w3']),
                                  jn_biases['jn_b3'])
        else:
            jn_embedding = tf.add_n(self._dspikes)

        # Synthetic network.
        syn_hidden1 = tf.nn.relu(
            tf.add(tf.matmul(text_embedding, syn_weights['syn_w1']),
                   syn_biases['syn_b1']))
        syn_hidden2 = tf.nn.relu(
            tf.add(tf.matmul(syn_hidden1, syn_weights['syn_w2']),
                   syn_biases['syn_b2']))
        syn_embedding = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']),
                               syn_biases['syn_b3'])
        self._syn_loss = tf.reduce_mean(
            tf.nn.l2_loss(tf.stop_gradient(jn_embedding) - syn_embedding))

        # Switch between Synthetic embedding and Joiner embedding.
        downstream_embedding = tf.cond(
            tf.equal(self._use_synthetic, tf.constant(True)),
            true_fn=lambda: tf.stop_gradient(syn_embedding),
            false_fn=lambda: jn_embedding)

        # Local embedding network.
        input_layer = tf.concat([text_embedding, downstream_embedding], axis=1)
        hidden_layer1 = tf.nn.relu(
            tf.add(tf.matmul(input_layer, l_weights['w1']), l_biases['b1']))
        hidden_layer2 = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer1, l_weights['w2']), l_biases['b2']))
        self._embedding = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer2, l_weights['w3']), l_biases['b3']))

        # Target: softmax cross entropy over local network embeddings.
        # Representation Weights & Sampled Softmax Loss.
        self._target_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=t_weights['t_w1'],
            biases=t_biases['t_b1'],
            labels=tf.reshape(label_tokens, [-1, 1]),
            inputs=self._embedding,
            num_sampled=self._hparams.n_sampled,
            num_classes=self._hparams.n_vocabulary,
            num_true=self._hparams.n_targets,
            sampled_values=None,
            remove_accidental_hits=True,
            partition_strategy='mod',
            name='sampled_softmax_loss',
            seed=None))

        # Optimizer: The optimizer for this component.
        optimizer = tf.compat.v1.train.AdamOptimizer(
            self._hparams.learning_rate)

        # Synthetic network grads from synthetic loss.
        self._syn_grads = optimizer.compute_gradients(loss=self._syn_loss,
                                                      var_list=syn_vars)

        # Downstream grads from upstream
        self._dgrads = optimizer.compute_gradients(loss=self._embedding,
                                                   var_list=self._dspikes,
                                                   grad_loss=self._egrads)

        # Local + joiner network grads from upstream.
        self._elgrads = optimizer.compute_gradients(loss=self._embedding,
                                                    var_list=l_vars + jn_vars,
                                                    grad_loss=self._egrads)

        # Downstream grads from target.
        self._tdgrads = optimizer.compute_gradients(loss=self._target_loss,
                                                    var_list=self._dspikes)

        # Local + joiner grads from target.
        self._tlgrads = optimizer.compute_gradients(loss=self._target_loss,
                                                    var_list=l_vars + jn_vars)


        # Train step for synthetic inputs.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Train step from embedding Local + joiner network grads.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Train step from target Local + joiner network grads.
        self._tstep = optimizer.apply_gradients(self._tlgrads)

        # FIM: Fishers information estimation.
        # Calculate contribution scores.
        # ∆Lij≈ ∑ gx·∆dj + 1/2N * ∆dj ∑ (gx∗gx)
        self._scores = []
        for i in range(self._hparams.n_children):
            delta_d = -self._dspikes[i]
            gx = self._tdgrads[i][0]
            g = tf.tensordot(delta_d, gx, axes=2)
            gxgx = tf.multiply(gx, gx)
            H = tf.tensordot(delta_d, gxgx, axes=2)
            score = tf.reduce_sum(g + H)
            self._scores.append(score)


    def _build_vocabulary(self):
        """ Parses the dummy corpus into a single sequential array.
        Each sentence is appended to each other. Also produces count dictionary
        for each word in the corpus.
        """

        # Read textfile.
        f = zipfile.ZipFile(self._hparams.corpus_path)
        for name in f.namelist():
            self._words = tf.compat.as_str(f.read(name)).split()
        f.close()

        counts = [('UNK', -1)]
        counts.extend(
            collections.Counter(self._words).most_common(self._hparams.n_vocabulary -
                                                        2))
        self._string_map = [c[0] for c in counts]

        logger.debug('Built Nucleus vocabulary.')
