import collections
from loguru import logger
import tensorflow as tf
import zipfile


class Nucleus():

    def __init__(self, config):
        self.config = config

        # Dataset zip file.
        self.filename = 'neurons/boltzmann/data/text8.zip'
        # Size of vocab embedding.
        self.vocabulary_size = 50000
        # Size of training batch.
        self.batch_size = 128
        # Dimension of the embedding vector.
        self.embedding_size = 128
        # Number of negative examples to sample.
        self.num_sampled = 64

        # Build Tokenizer vocabulary.
        self.build_vocabulary()

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            token_embedding = self.tokenization_graph()
            downstream_embeddings = self.downstream_graph()
            self.nucleus_graph(token_embedding, downstream_embeddings)

        # Create TF Session.
        self.session = tf.compat.v1.Session(graph=self.graph)

        # Init tables and vars.
        self.session.run(self.var_init)

        self.session.run(self.table_init)


    def distill(self, uspikes):

        # Build Feeds dictionary.
        feeds = {self.text_placeholder: uspikes,
                 self.is_distill: True}

        # Build Fetches dictionary.
        fetches = {'output': self.output}

        # Run graph.
        run_output = self.session.run(fetches, feeds)

        # Return spikes.
        return run_output['output']


    def grade(self, ugrades, uspikes, dspikes):

        # Build Feeds dictionary.
        feeds = {}
        feeds[self.text_placeholder] = uspikes
        feeds[self.output_grad] = ugrades
        for i in range(self.config.k):
            feeds[self.dspikes[i]] = dspikes[i]

        fetches = {}
        fetches['lgrads'] = self.gradient_values
        for i in range(self.config.k):
            fetches["dgrads" + str(i)] = self.downstream_grads[i]

        # Run graph.
        run_output = self.session.run(fetches, feeds)

        # Return spikes.
        return [run_output["dgrads" + str(i)] for i in range(self.config.k)
               ], run_output['lgrads']

    def learn(self, gradients):

        # Build Feeds dictionary.
        # Feed batch of gradients.
        feeds = {}
        for i, grad_var in enumerate(gradients):
            feeds[self.placeholder_gradients[i][0]] = gradients[i]

        # Fetches. Call apply gradients.
        fetches = {}
        fetches['step'] = self.step

        # Run graph. No output.
        self.session.run(fetches, feeds)

    def spike(self, uspikes, dspikes):

        # Build Feeds dictionary.
        feeds = {self.text_placeholder: uspikes}
        for i in range(self.config.k):
            feeds[self.dspikes[i]] = dspikes[i]

        # Build Fetches dictionary.
        fetches = {'output': self.output}

        # Run graph.
        run_output = self.session.run(fetches, feeds)

        # Return spikes.
        return run_output['output']

    def tokenize(self, text):
        # Build Feeds dictionary.
        feeds = {self.text_placeholder: text}

        # Build Fetches dictionary.
        fetches = {'token_embedding': self.token_embedding}

        # Run graph.
        run_output = self.session.run(fetches, feeds)

        # Return spikes.
        return run_output['token_embedding']

    def tokenization_graph(self):
        # Text placeholder. Text should be unicoded encoded strings. Here
        # we are treating them as single words.
        # TODO(const): tokenize larger strings.
        self.text_placeholder = tf.compat.v1.placeholder(
            tf.string, shape=[None, 1], name="text_placeholder")
        text = tf.reshape(self.text_placeholder, [-1])

        # Tokenization with loopup table. This is the simplest form of
        # tokenization which simply looks up the word in a table to retrieve a
        # 1 x vocabulary sized vector.
        # string map, is a list of strings ordered by count.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(self.string_map),
            num_oov_buckets=1,
            default_value=0)

        # Apply tokenizer lookup.
        input_tokens = vocabulary_table.lookup(input_text)

        # Token embedding matrix is a matrix of vectors. During lookup we pull
        # the vector corresponding to the 1-hot encoded vector from the
        # vocabulary table.
        token_embedding_matrix = tf.Variable(
            tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0,
                              1.0))

        # Apply table lookup to retrieve the embedding.
        embedding = tf.nn.embedding_lookup(embedding_matrix, input_tokens)

        # reshape and return.
        self.token_embedding = tf.reshape(embedding, [-1, self.embedding_size])
        return self.token_embedding


    def downstream_spikes(self):
        # Placeholders for downstream spikes.
        self.dspikes = []
        for i in range(self.config.k):
            downstream_spikes = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, self.embedding_size],
                name="dspikes_placeholder" + str(i))
            self.dspikes.append(downstream_spikes)


        # Build downstream grad tensors.
        self.downstream_grads = []
        for i in range(self.config.k):
            dspikes_grad = tf.gradients(xs=[self.dspikes[i]],
                                        ys=self.output,
                                        grad_ys=self.output_grad,
                                        name="dgrads" + str(i))
            self.downstream_grads.append(dspikes_grad)

        return self.dspikes



    def student(self, token_embedding):
        '''Builds the student model, returns the students's embedding'''
        weights = {
            'w1':
                tf.Variable(
                    tf.truncated_normal([config.n_inputs, config.s_hidden1],
                                        stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.truncated_normal([config.s_hidden1, config.s_hidden2],
                                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.truncated_normal([config.s_hidden2, config.n_embedding],
                                        stddev=0.1)),
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[config.s_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[config.s_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[config.n_embedding])),
        }

        layer_1 = tf.nn.relu(
            tf.add(tf.matmul(token_embedding, weights['w1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
                                    biases['b2']))
        student_embedding = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
        return student_embedding



    def teacher(self, token_embedding, student_embedding, hparams):
        '''Builds the teacher model, returns the teacher's embedding'''

        teacher_inputs = tf.concat([token_embedding, student_embedding], axis=1)
        n_teacher_inputs = hparams.n_inputs + hparams.n_embedding

        weights = {
            'w1':
                tf.Variable(
                    tf.truncated_normal([n_teacher_inputs, hparams.t_hidden1],
                                        stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.truncated_normal([hparams.t_hidden1, hparams.t_hidden2],
                                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.truncated_normal([hparams.t_hidden2, hparams.n_embedding],
                                        stddev=0.1)),
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[hparams.n_embedding])),
        }

        layer_1 = tf.nn.relu(
            tf.add(tf.matmul(teacher_inputs, weights['w1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
                                    biases['b2']))
        teacher_embedding = tf.nn.relu(
            tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))

        return teacher_embedding

    def classroom(self, downstream_spikes):
        return tf.add_n(downstream_spikes)

    def loss(self, embedding):
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.alpha)
        gradients = self.optimizer.compute_gradients(loss=self.output,
                                                     grad_loss=self.output_grad)

        # Build gradient placeholders for the Learn step.
        self.gradient_values = []
        self.placeholder_gradients = []
        for gradient_variable in gradients:
            grad_placeholder = tf.placeholder(
                'float', shape=gradient_variable[1].get_shape())
            self.gradient_values.append(gradient_variable[1])
            self.placeholder_gradients.append(
                (grad_placeholder, gradient_variable[1]))

        self.step = self.optimizer.apply_gradients(self.placeholder_gradients)
        return self.step

    def logits(self, embedding, hparams):
        '''Calculates the teacher and student logits from embeddings.'''
        w = tf.Variable(
            tf.truncated_normal([hparams.n_embedding, hparams.n_targets],
                                stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[hparams.n_targets])),
        logits = tf.add(tf.matmul(embedding, w), b)
        return logits

    def target_loss(self, logits, targets):
        '''Calculates the target loss w.r.t a set of logits.'''
        target_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets,
                                                       logits=logits))
        return target_loss

    def distillation_loss(self, student_embedding, teacher_embedding):
        '''Calculates the distilled loss between the teacher and student embedding'''
        distillation_loss = tf.reduce_mean(
            tf.nn.l2_loss(tf.stop_gradient(teacher_embedding) - student_embedding))
        return distillation_loss

    def nucleus_graph(self, downstream_embeddings, token_embedding):

        self.classroom_embedding = self.classroom(downstream_embeddings)

        self.student_embedding = self.student(token_embedding)

        self.distillation_loss = self.distillation_loss(self.student_embedding, self.classroom_embedding)

        self.teacher_embedding = self.teacher(token_embedding, self.student_embedding)

        self.logits = self.logits(self.teacher_embedding)

        self.target_loss = self.target_loss(self.logits)

        self.step = self.optimizer(self.target_loss, self.distillation_loss)

        # Init vars.
        self.var_init = tf.compat.v1.global_variables_initializer()
        self.table_init = tf.compat.v1.tables_initializer(
            name='init_all_tables')

        logger.debug('Built Nucleus graph.')

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
        counts.extend(
            collections.Counter(self.words).most_common(self.vocabulary_size -
                                                        2))
        self.string_map = [c[0] for c in counts]

        logger.debug('Built Nucleus vocabulary.')
