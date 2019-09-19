import bittensor

from config import Config

from loguru import logger
import os
import time
import tensorflow as tf
import tensor2tensor.data_generators.cola  as cola
import threading
import urllib.request
import zipfile

# URL for downloading COLA data.
CoLA_URL ="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"

# Downloads the CoLA dataset.
def download_and_extract_cola():
    logger.info("Downloading and extracting CoLA into neurons/CoLA/data")
    data_file = "CoLA.zip"
    urllib.request.urlretrieve(CoLA_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall("neurons/CoLA/data")
    os.remove(data_file)
    logger.info("\tCompleted!")

class Dendrite():
    def __init__(self):
        pass

    def spike(self, inputs):
        return tf.Variable(tf.random_uniform([1,EMBEDDING_SIZE]))

EMBEDDING_SIZE = 128

class Neuron():

    def __init__(self, config, dendrite):
        self._config = config
        self._dendrite = dendrite
        self._cola = cola.Cola()
        self._cola_generator = self._cola.example_generator('neurons/CoLA/data/CoLA/dev.tsv')

        # Master thread.
        self._master_thread = threading.Thread(target=self._main)
        self._master_thread.setDaemon(True)
        self._running = False

        # Build graph.
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._build_preprocess_graph(self._graph)
            self._build_training_graph(self._graph)
            self._init = tf.compat.v1.global_variables_initializer()

    def _build_preprocess_graph(self, graph):
        self._inputs = tf.compat.v1.placeholder(tf.string, shape=[1, 1], name="inputs")
        embeddings = self._dendrite.spike(self._inputs)
        self._queue = tf.queue.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=[1, EMBEDDING_SIZE])
        self._enqueue_op = self._queue.enqueue(embeddings)

    def _build_training_graph(self, graph):
        # Inputs.
        self._global_step = tf.compat.v1.train.create_global_step()
        next_embeddings = self._queue.dequeue()

        # Layer 1
        w1 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([EMBEDDING_SIZE]))
        h1 = tf.sigmoid(tf.matmul(next_embeddings, w1) + b1)

        # Layer 2.
        w2 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, 1], -1.0, 1.0))
        b2 = tf.Variable(tf.zeros([1]))
        y = tf.sigmoid(tf.matmul(h1, w2) + b2)

        # Loss calculation.
        self._label = tf.compat.v1.placeholder(tf.float32, shape=[1, 1], name="label")
        self._loss = tf.losses.log_loss(self._label, y)
        self._step = tf.compat.v1.train.AdagradOptimizer(1.0).minimize(self._loss, global_step=self._global_step)


    def start(self):
        self._running = True
        self._master_thread.start()

    def stop(self):
        self._running = False
        if self._coord:
            self._coord.request_stop()
        self._master_thread.join()

    def _main(self):
        logger.debug('Started Nucleus training.')
        with self._session:
            try:
                # Set up threading coordinator.
                self._coord = tf.train.Coordinator()

                # Initialize the graph.
                self._session.run(self._init)

                # Create and start the training and preprocessing threads.
                preproccess_thread = threading.Thread(target=self._preprocessing_loop)
                training_thread = threading.Thread(target=self._training_loop)
                preproccess_thread.setDaemon(True)
                training_thread.setDaemon(True)
                preproccess_thread.start()
                training_thread.start()

                # Wait on threads.
                self._coord.join([preproccess_thread, training_thread])

            except Exception as e:
                # Stop on exception.
                self._coord.request_stop(e)
                self._coord.join([preproccess_thread, training_thread])
                logger.error(e)

            finally:
                # Stop on all other exits.
                self._coord.request_stop()
                self._coord.join([preproccess_thread, training_thread])
                logger.debug('Stopped Nucleus training.')

    def _preprocessing_loop(self):
        try:
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:
                    next_example = next(self._cola_generator)
                    run_output = self._session.run(  fetches=[self._enqueue_op],
                                                    feed_dict={self._inputs: next_example['inputs'],
                                                               self._label: next_example['label']})
        except Exception as e:
            self._coord.request_stop(e)

    def _training_loop(self):
        try:
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:
                    run_output = self._session.run(  fetches={self._step},
                                                    feed_dict={})
        except Exception as e:
            self._coord.request_stop(e)
        logger.info('Stopped training thread.')




def main():

    # Download the data.
    download_and_extract_cola()

    config = None

    dendrite = Dendrite()

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
        logger.error('Neuron stopped with interrupt on error: '+ str(e))
        tear_down(config, dendrite, neuron)

if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
