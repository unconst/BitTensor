import collections
import math
from matplotlib import pylab
import numpy as np
import random
from sklearn.manifold import TSNE
import tensorflow as tf
import time
import threading
import zipfile

def read_data(filename):
    """Reads a text.zip file and splits the text into words.
    Args:
      filename: Zipped text file to read.
    Returns:
      A list of strings.
    """
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name)).split()
    f.close()

def build_dataset(words, vocabulary_size):
    """Parses a list of words into a NLP dataset.
    Args:
        words: A list of words from loaded text.
    Returns:
        data: A list of embbeddings for each index.
        count: A count of all UNK words.
        dictionary: A map from word to it's embedding index.
        reversed_dictionary: A map from embedding index to word.
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_random_batch(data, batch_size):
    """ Generates a random batch example skip gram tuples (word, word + 1)
    Args:
        data: A list of embedding indices.
        batch_size: The size of the batch.
    Returns:
        batch: A list of input vocab indicies.
        labels: A list of label vocab indicies.
    """
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        index = random.randint(0, len(data) - 1)
        batch[i] = data[index]
        labels[i] = data[index]
    return batch, labels



_ONE_DAY_IN_SECONDS = 60*60*24

class Neuron():
    def __init__(self):
        self.train_thread = threading.Thread(target=self._train)
        self.train_thread.setDaemon(True)
        self.running = False
        self.mutex = threading.Lock()

        # Dataset zip file.
        filename = 'text8.zip'
        # Size of vocab embedding.
        vocabulary_size = 50000 # Size of vocab embedding.
        # Size of training batch.
        self.batch_size = 128 # Size of training batch.
        # Dimension of the embedding vector.
        embedding_size = 128
        # Number of TSNE points.
        num_points = 400
        # Number of negative examples to sample.
        num_sampled = 64

        # 1. Read file text2.zip into a words
        print ('Loading Vocab...')
        print ('    file %s' % filename)
        words = read_data(filename)
        print ('    size %d' % len(words))
        print ('done. \n')

        # 2. Create a vocabulary dataset.
        self.data, count, self.dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
        print ('Building dataset...')
        print ('    e.g.')
        print ('        words:', words[:5])
        print ('        indices',  self.data[:5])
        del words  # Hint to reduce memory.
        print ('done. \n')

        # 3. Test generate batch function.
        batch, labels = generate_random_batch(self.data, 5)
        print ('Testing batch gen...')
        print ('    e.g.:')
        print ('        batch:', [reverse_dictionary[i] for i in batch.tolist()])
        print ('        label:', [reverse_dictionary[i] for i in labels.flatten().tolist()])
        print ('done. \n')

        # 4. Build Graph.
        print ('Building graph...')
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):

            # Input data.
            self.train_dataset = tf.reshape((tf.placeholder(tf.int32, shape=[None, 1])), [-1])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

            # Variables.
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

            # Embedding Weights
            weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Look up embeddings for inputs.
            self.embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)

            # Sampled Softmax Loss.
            batch_loss = tf.nn.sampled_softmax_loss(
                weights=weights,
                biases=biases,
                labels=self.train_labels,
                inputs=self.embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size,
                num_true=1,
                sampled_values=None,
                remove_accidental_hits=True,
                partition_strategy='mod',
                name='sampled_softmax_loss',
                seed=None)

            # Average loss.
            self.loss = tf.reduce_mean(batch_loss)

            # Optimizer.
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

        self.session = tf.Session(graph=self.graph)

    def start(self):
        self.running = True
        self.train_thread.start()

    def stop(self):
        self.running = False
        self.train_thread.join()

    def spike(self, batch):
        with self.mutex:
            # Encode batch.
            batch_data = np.ndarray(shape=(len(batch)), dtype=np.int32)
            for i in range(len(batch)):
                word = batch[i]
                index = self.dictionary[word]
                batch_data[i] = self.data[index]


            feed_dict = {self.train_dataset : batch_data}
            return self.session.run(self.embed, feed_dict=feed_dict)


    def _train(self):
        print ('Training ...')
        with self.session:

            # Init.
            tf.global_variables_initializer().run()

            # Train loop.
            average_loss = 0
            step = -1
            while self.running:
                with self.mutex:
                    step += 1

                    # Train Step.
                    batch_data, batch_labels = generate_random_batch(self.data, self.batch_size)
                    feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
                    _, l = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    average_loss += l

                    # Progress notification.
                    if step % 2000 == 1 and step > 2000:
                        print('     Average loss at step %d: %f' % (step, average_loss/2000))
                        average_loss = 0


        print ('done. \n')





