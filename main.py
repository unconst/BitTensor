import collections
import math
from matplotlib import pylab
import numpy as np
import random
from sklearn.manifold import TSNE
import tensorflow as tf
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
        data: A list of embbedding indices for each word in words.
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
        index = random.randint(0, len(data) - 2)
        batch[i] = data[index]
        labels[i] = data[index + 1]
    return batch, labels


def main():

    # 0. Args
    # Dataset zip file.
    filename = 'text8.zip'
    # Size of vocab embedding.
    vocabulary_size = 50000 # Size of vocab embedding.
    # Size of training batch.
    batch_size = 128 # Size of training batch.
    # Dimension of the embedding vector.
    embedding_size = 128
    # Number of TSNE points.
    num_points = 400
    # Number of negative examples to sample.
    num_sampled = 64
    # Number of training steps.
    num_steps = 10000

    # 1. Read file text2.zip into a words
    print ('Loading Vocab...')
    print ('    file %s' % filename)
    words = read_data(filename)
    print ('    size %d' % len(words))
    print ('done. \n')

    # 2. Create a vocabulary dataset.
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print ('Building dataset...')
    print ('    e.g.')
    print ('        words:', words[:5])
    print ('        indices',  data[:5])
    del words  # Hint to reduce memory.
    print ('done. \n')

    # 3. Test generate batch function.
    batch, labels = generate_random_batch(data, 5)
    print ('Testing batch gen...')
    print ('    e.g.:')
    print ('        batch:', [reverse_dictionary[i] for i in batch.tolist()])
    print ('        label:', [reverse_dictionary[i] for i in labels.flatten().tolist()])
    print ('done. \n')

    # 4. Build Graph.
    print ('Building graph...')
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Variables.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # Weights.
        random_normal = tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
        weights = tf.Variable(random_normal)
        biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Embeddings lookup.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)

        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights, biases, train_labels, embed, num_sampled, vocabulary_size))

        # Optimizer.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Normalized Embeddings for TSNE.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
    print ('done. \n')

    # 5. Train.
    print ('Training ...')
    with tf.Session(graph=graph) as session:

        # Init.
        tf.global_variables_initializer().run()

        # Train loop.
        average_loss = 0
        for step in range(num_steps):

            # Train Step.
            batch_data, batch_labels = generate_random_batch(data, batch_size)
            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l

            # Progress notification.
            if step % 2000 == 1:
                print('     Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

        final_embeddings = normalized_embeddings.eval()
    print ('done. \n')


    # 6. Visualize Embeddings using TSNE.
    print ('TSNE...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(10,10))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom')
        pylab.show()

    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
    print ('    plotting.')
    plot(two_d_embeddings, words)
    print ('done. \n')

if __name__ == "__main__":
    main()
