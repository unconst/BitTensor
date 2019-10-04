import bittensor

from config import Config
from dendrite import Dendrite
from metagraph import Metagraph

from itertools import cycle
from loguru import logger
import numpy as np
import os
import pickle
import queue
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
    if os.path.exists("neurons/CoLA/data/CoLA"):
        return
    logger.info("Downloading and extracting CoLA into neurons/CoLA/data")
    data_file = "CoLA.zip"
    urllib.request.urlretrieve(CoLA_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall("neurons/CoLA/data")
    os.remove(data_file)
    logger.info("\tCompleted!")


def next_batch(size, generator):
    # Build batch.
    batch_text = []
    batch_labels = []
    for _ in range(size):
        sample = next(generator)
        batch_text.append([sample['inputs']])
        batch_labels.append([sample['label']])
    batch_text = np.array(batch_text)
    batch_labels = np.array(batch_labels)
    return batch_text, batch_labels


EMBEDDING_SIZE = 128


class Neuron():

    def __init__(self, config, dendrite):
        self._config = config
        self._dendrite = dendrite
        self._cola = cola.Cola()
        self._generator = cycle(
            self._cola.example_generator('neurons/CoLA/data/CoLA/train.tsv'))

        # batch_size
        self._batch_size = self._config.batch_size

        # Master thread.
        self._master_thread = threading.Thread(target=self._main)
        self._master_thread.setDaemon(True)
        self._running = False

        # preprocessing queue.
        self._queue = queue.Queue(maxsize=1)

        # Build graph.
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._build_training_graph()
            self._init = tf.compat.v1.global_variables_initializer()

        # Summary writer for tensorboard.
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            self._config.logdir, self._graph)

    def _preprocessing_loop(self):
        nounce = 0
        try:
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:

                    # Create a batch of examples from our data generator.
                    text, labels = next_batch(self._batch_size, self._generator)

                    # Query Bittensor Network
                    futures = self._dendrite.spike(str(nounce), text)

                    # Put item into the queue. If optional args block is true and
                    # timeout is None (the default), block if necessary until a
                    # free slot is available. If timeout is a positive number,
                    # it blocks at most timeout seconds and raises the Full exception
                    # if no free slot was available within that time. Otherwise
                    # (block is false), put an item on the queue if a free slot
                    # is immediately available, else raise the Full exception
                    # (timeout is ignored in that case).
                    self._queue.put(
                        {
                            'nounce': str(nounce),
                            'text': text,
                            'labels': labels,
                            'futures': futures
                        },
                        timeout=5,
                        block=True)

                    nounce += 1

        except Exception as e:
            logger.error(e)
            self._coord.request_stop(e)

    def _build_training_graph(self):
        # Inputs.
        # Global step.
        self._global_step = tf.compat.v1.train.create_global_step()
        self._nounce = tf.compat.v1.placeholder(tf.string, shape=[])
        self._text = tf.compat.v1.placeholder(tf.string,
                                              shape=[self._batch_size, 1])
        self._labels = tf.compat.v1.placeholder(tf.float32,
                                                shape=[self._batch_size, 1])
        self._inputs = []
        for i in range(self._config.k):
            input_i = tf.compat.v1.placeholder(
                tf.float32, shape=[self._batch_size, EMBEDDING_SIZE])
            self._inputs.append(input_i)

        # Layer 1
        input_layer = tf.concat(self._inputs, axis=1)
        w1_shape = [EMBEDDING_SIZE * self._config.k, EMBEDDING_SIZE]
        w1 = tf.Variable(tf.random.uniform(w1_shape, -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([EMBEDDING_SIZE]))
        h1 = tf.sigmoid(tf.matmul(input_layer, w1) + b1)

        # Layer 2.
        w2 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, 1], -1.0, 1.0))
        b2 = tf.Variable(tf.zeros([1]))
        y = tf.sigmoid(tf.matmul(h1, w2) + b2)

        # Loss calculation.
        self._loss = tf.losses.log_loss(self._labels, y)
        tf.compat.v1.summary.scalar('loss', self._loss)

        # Optimizer and downstream gradients.
        optimizer = tf.compat.v1.train.AdagradOptimizer(self._config.alpha)
        grads_and_vars = optimizer.compute_gradients(self._loss,
                                                     var_list=self._inputs)
        self._gradients = [grad for (grad, var) in grads_and_vars]
        self._train = optimizer.minimize(self._loss)

        # Calculate contribution scores.
        # ∆Lij≈ ∑ gx·∆dj + 1/2N * ∆dj ∑ (gx∗gx)
        self._deltaLij = []
        for i, gx in enumerate(self._gradients):
            delta_d = -self._inputs[i]
            g = tf.tensordot(delta_d, gx, axes=2)
            gxgx = tf.multiply(gx, gx)
            H = tf.tensordot(delta_d, gxgx, axes=2)
            score = tf.reduce_sum(g + H)
            self._deltaLij.append(score)

    def _training_loop(self):
        try:
            mavg_deltaLij = [0.0 for _ in range(self._config.k)]
            with self._coord.stop_on_exception():
                while not self._coord.should_stop() and self._running:

                    # Pull next sample from preprocessing queue.
                    example = self._queue.get(block=True, timeout=5)

                    # Unpack.
                    nounce = example['nounce']
                    text = example['text']
                    labels = example['labels']
                    futures = example['futures']

                    # Build feeds.
                    feeds = {
                        self._nounce: nounce,
                        self._text: text,
                        self._labels: labels
                    }

                    # Fill feeds from passed call futures. Iterates through
                    # each channel checking done complete on futures. If they
                    # are filled we add them to the feed dict, if they are none
                    # we fill the feed dict with zeros.
                    ttl = 1.0
                    start = time.time()
                    is_done = [False for _ in futures]
                    is_filled = [False for _ in futures]
                    while True:
                        for i, channel in enumerate(self._inputs):
                            # Init as zeros.
                            feeds[channel] = np.zeros(
                                (self._batch_size, EMBEDDING_SIZE))

                            # Check already done.
                            if is_done[i]:
                                continue

                            # Check nil.
                            elif futures[i] == None:
                                is_done[i] = True

                            # Check result.
                            elif futures[i].done():
                                is_done[i] = True
                                try:
                                    response = futures[i].result()
                                    dspikes = pickle.loads(response.payload)
                                    feeds[channel] = dspikes.reshape(
                                        self._batch_size, EMBEDDING_SIZE)
                                    is_filled[i] = True
                                except:
                                    pass

                            # Check ttl.
                            elif (time.time() - start) > ttl:
                                is_done[i] = True

                        # Break fill when all are done or timedout.
                        if all(is_done):
                            break

                    # Build fetches.
                    fetches = {
                        'train': self._train,
                        'loss': self._loss,
                        'step': self._global_step,
                        'gradients': self._gradients,
                        'deltaLij': self._deltaLij
                    }

                    # # Run Training graph.
                    _output = self._session.run(fetches=fetches,
                                                feed_dict=feeds)

                    # Unpack.
                    loss = _output['loss']
                    step = _output['step']
                    gradients = _output['gradients']

                    # Update moving deltaij
                    deltaLij = _output['deltaLij']
                    for i in range(len(deltaLij)):
                        mavg_deltaLij[i] = (0.95) * mavg_deltaLij[i] + (
                            0.05 * abs(deltaLij[i]))

                    # Train network.
                    self._dendrite.grad(nounce, text, gradients)

                    # Write summaries.
                    logger.info('step: {} loss: {} delatLij: {}', nounce,
                                ("%.4f" % loss),
                                [("%.4f" % dl) for dl in mavg_deltaLij])

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
