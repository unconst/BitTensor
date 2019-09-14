import collections
from loguru import logger
import tensorflow as tf
import zipfile

class Nucleus():
    def __init__(self, config):
        self.config = config

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

        # Build Tokenizer vocabulary.
        self.build_vocabulary()

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.build_graph()

        # Create TF Session.
        self.session = tf.compat.v1.Session(graph=self.graph)

        # Init tables and vars.
        self.session.run(self.var_init)

        self.session.run(self.table_init)


    def spike(self, uspikes, dspikes):

        # Build Feeds dictionary.
        feeds = { self.text_placeholder: uspikes }
        for i in self.config.k:
            feeds['dspikes_placeholder' + str(i)] = dspikes[i]

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
        for i in self.config.k:
            feeds['dspikes_placeholder' + str(i)] = dspikes[i]


        fetches = {}
        fetches['local_gradients'] = self.gradients
        for i in self.config.k:
            fetches["dgrads" + str(i)] = self.downstream_grads[i]

        # Run graph.
        run_output = self.session.run(fetches, feeds)

        # Return spikes.
        return run_output['output']


    def train(self, gradients):

        # Build Feeds dictionary.
        # Feed batch of gradients.
        feeds = {}
        feeds[self.gradients] = gradients

        # Fetches. Call apply gradients.
        fetches = {}
        fetches['step'] = self.step

        # Run graph. No output.
        self.session.run(fetches, feeds)

    def build_graph (self):

        # Text input placeholder.
        self.text_placeholder = tf.compat.v1.placeholder(tf.string, shape=[None, 1], name="text_placeholder")
        input_text = tf.reshape(self.text_placeholder, [-1])

        # Tokenization.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.string_map), num_oov_buckets=1, default_value=0)
        input_tokens = vocabulary_table.lookup(input_text)

        # Token spikes.
        embedding_matrix = tf.Variable(tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        token_spikes = tf.nn.embedding_lookup(embedding_matrix, input_tokens)
        token_spikes = tf.reshape(token_spikes, [-1, self.embedding_size])

        # Placeholders for downstream spikes.
        all_dspikes = []
        for i in range(self.config.k):
            downstream_spikes = tf.compat.v1.placeholder(tf.float32, shape=[None, self.embedding_size], name="dspikes_placeholder" + str(i))
            all_dspikes.append(downstream_spikes)

        # activation_spikes = [None, embedding_size * (self.config.k + 1)]
        self.activation_size = self.embedding_size * (self.config.k + 1)
        self.activation_spikes = tf.concat([token_spikes] + all_dspikes, axis = 1)

        # Layer 1.
        w1 = tf.Variable(tf.random.uniform([self.activation_size, self.embedding_size], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([self.embedding_size]))
        local_spikes = tf.sigmoid(tf.matmul(self.activation_spikes, w1) + b1)

        # Representation. Output Spikes,
        self.output = tf.identity(local_spikes, name="output")

        # Upstream gradient placeholder.
        self.output_grad = tf.placeholder(tf.float32, [None, self.embedding_size])

        # Build downstream grad tensors.
        self.downstream_grads = []
        for i in range(self.config.k):
            dspikes = all_dspikes[i]
            dspikes_grad = tf.gradients(xs=[dspikes], ys=self.output,  grad_ys=self.output_grad, name="dgrads" + str(i))
            self.downstream_grads.append(dspikes_grad)

        # Build optimizer.
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.alpha)
        self.gradients = self.optimizer.compute_gradients(loss=self.output, grad_loss=self.output_grad)
        self.step = self.optimizer.apply_gradients(self.gradients)

        # Init vars.
        self.var_init = tf.compat.v1.global_variables_initializer()
        self.table_init = tf.compat.v1.tables_initializer(name='init_all_tables')

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
        counts.extend(collections.Counter(self.words).most_common(self.vocabulary_size - 2))
        self.string_map = [c[0] for c in counts]

        logger.debug('Built Nucleus vocabulary.')
