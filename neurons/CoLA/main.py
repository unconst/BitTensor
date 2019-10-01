import bittensor

from config import Config
from dendrite import Dendrite
from metagraph import Metagraph

from itertools import cycle
from loguru import logger
import numpy as np
import os
import time
import tensorflow as tf
import tensor2tensor.data_generators.cola as cola
import threading
import urllib.request
import zipfile

# URL for downloading COLA data.
CoLA_URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"


# Downloads the CoLA dataset.
def download_and_extract_cola():
    logger.info("Downloading and extracting CoLA into neurons/CoLA/data")
    data_file = "CoLA.zip"
    urllib.request.urlretrieve(CoLA_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall("neurons/CoLA/data")
    os.remove(data_file)
    logger.info("\tCompleted!")


EMBEDDING_SIZE = 128


class Neuron():

    def __init__(self, config, dendrite):
        self._config = config
        self._dendrite = dendrite
        self._cola = cola.Cola()
        self._cola_generator = cycle(
            self._cola.example_generator('neurons/CoLA/data/CoLA/train.tsv'))

        # batch_size
        self._batch_size = self._config.batch_size

        # Master thread.
        self._master_thread = threading.Thread(target=self._main)
        self._master_thread.setDaemon(True)
        self._running = False

        # Build graph.
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._build_preprocess_graph()
            self._build_training_graph()
            self._init = tf.compat.v1.global_variables_initializer()

    def _build_preprocess_graph(self):
        # Inputs
        self._batch_nounce = tf.compat.v1.placeholder(tf.string, shape=[])
        self._batch_embeddings = tf.compat.v1.placeholder(tf.float32, shape=[self._config.k, self._batch_size, EMBEDDING_SIZE])
        self._batch_text = tf.compat.v1.placeholder(tf.string, shape=[self._batch_size, 1])
        self._batch_labels = tf.compat.v1.placeholder(tf.float32, shape=[self._batch_size, 1])

        # Build Queue..
        dtypes = [tf.string,
                  tf.float32,
                  tf.string,
                  tf.float32]
        shapes = [  [],
                    [self._config.k, self._batch_size, EMBEDDING_SIZE],
                    [self._batch_size, 1],
                    [self._batch_size, 1]]
        self._queue = tf.queue.FIFOQueue(capacity=100,
                                         dtypes=dtypes,
                                         shapes=shapes)

        # Enqueue.
        self._enqueue_step = self._queue.enqueue([self._batch_nounce,
                                                 self._batch_embeddings,
                                                 self._batch_text,
                                                 self._batch_labels])

    def _preprocessing_loop(self):
        nounce = 0
        try:
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:
                    # Build batch.
                    batch_text = []
                    batch_labels = []
                    for _ in range(self._batch_size):
                        sample = next(self._cola_generator)
                        batch_text.append([sample['inputs']])
                        batch_labels.append([sample['label']])
                    batch_text = np.array(batch_text)
                    batch_labels = np.array(batch_labels)

                    # Query Bittensor Network
                    logger.info('preprocess nounce: {}', str(nounce).encode())
                    embeddings = self._dendrite.spike(str(nounce).encode(), batch_text)

                    # Build Feeds and fetches
                    feeds = {
                        self._batch_nounce: str(nounce),
                        self._batch_embeddings: embeddings,
                        self._batch_text: batch_text,
                        self._batch_labels: batch_labels
                    }
                    fetches = [self._enqueue_step]

                    # Run preprocessing
                    self._session.run(fetches, feeds)
                    nounce += 1

        except Exception as e:
            logger.error(e)
            self._coord.request_stop(e)


    def _build_training_graph(self):
        # Inputs.
        self.nounce, self.embeddings, self.text, labels = self._queue.dequeue()
        self.embeddings = tf.split(self.embeddings, self._config.k, 0)

        # Layer 1
        embeddings = tf.reshape(
            self.embeddings, [self._batch_size, EMBEDDING_SIZE * self._config.k])
        w1 = tf.Variable(
            tf.random.uniform([EMBEDDING_SIZE * self._config.k, EMBEDDING_SIZE],
                              -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([EMBEDDING_SIZE]))
        h1 = tf.sigmoid(tf.matmul(embeddings, w1) + b1)

        # Layer 2.
        w2 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, 1], -1.0, 1.0))
        b2 = tf.Variable(tf.zeros([1]))
        y = tf.sigmoid(tf.matmul(h1, w2) + b2)

        # Loss calculation.
        self._loss = tf.losses.log_loss(labels, y)

        # Optimizer and downstream gradients.
        optimizer = tf.compat.v1.train.AdagradOptimizer(self._config.alpha)
        self.embedding_grads = optimizer.compute_gradients(self._loss, var_list=self.embeddings)
        self._step = optimizer.minimize(self._loss)


    def _training_loop(self):
        try:
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:
                    # Build graph fetches.
                    fetches = {'loss': self._loss,
                               'step': self._step,
                               'grads': self.embedding_grads,
                               'text': self.text,
                               'embeddings': self.embeddings,
                               'nounce': self.nounce
                               }

                    # Run training graph.
                    run_output = self._session.run(fetches)
                    logger.info('train nounce: {} loss: {}', run_output['nounce'], run_output['loss'])

                    # Train Bittensor network.
                    self._dendrite.grad(run_output['nounce'],
                                        run_output['text'],
                                        run_output['grads'])

        except Exception as e:
            logger.error(e)
            self._coord.request_stop(e)
        logger.info('Stopped training thread.')

    def _main(self):
        logger.debug('Started Nucleus training.')
        with self._session:
            try:
                # Set up threading coordinator.
                self._coord = tf.train.Coordinator()

                # Initialize the graph.
                self._session.run(self._init)

                # Create and start the training and preprocessing threads.
                preproccess_thread = threading.Thread(
                    target=self._preprocessing_loop)
                training_thread = threading.Thread(target=self._training_loop)
                preproccess_thread.setDaemon(True)
                training_thread.setDaemon(True)
                preproccess_thread.start()
                training_thread.start()

                # Wait on threads.
                self._coord.join([preproccess_thread, training_thread])

            except Exception as e:
                # Stop on exception.
                logger.error(e)
                self._coord.request_stop(e)
                self._coord.join([preproccess_thread, training_thread])

            finally:
                # Stop on all other exits.
                self._coord.request_stop()
                self._coord.join([preproccess_thread, training_thread])
                logger.debug('Stopped Nucleus training.')

    def start(self):
        logger.info('start')
        self._running = True
        self._master_thread.start()

    def stop(self):
        logger.info('stop')
        self._running = False
        if self._coord:
            self._coord.request_stop()
        self._master_thread.join()


def main():

    # Download the data.
    download_and_extract_cola()

    config = Config()

    metagraph = Metagraph(config)

    dendrite = Dendrite(config, metagraph)

    neuron = Neuron(config, dendrite)

    neuron.start()

    def tear_down(_config, _dendrite, _neuron):
        logger.debug('tear down.')
        _neuron.stop()
        del _neuron
        del _dendrite
        del _config

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(config, dendrite, neuron)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: ' + str(e))
        tear_down(config, dendrite, neuron)


if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
