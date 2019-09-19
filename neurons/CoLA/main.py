import bittensor

from config import Config

from loguru import logger
import os
import time
import tensor2tensor.data_generators.cola  as cola
import urllib.request
import zipfile

# URL for downloading COLA data.
CoLA_URL ="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"

class Dendrite():
    def __init__(self):
        pass

EMBEDDING_SIZE = 128

class Neuron():

    def __init__(self, config, dendrite):
        self.config = config
        self.dendrite = dendrite
        self.cola = cola.Cola()
        self.cola_generator = self.cola.example_generator('neurons/CoLA/data/CoLA/dev.tsv')
        self._build_preprocess_graph()
        self._build_training_graph()

    def _build_preprocess_graph(self):
        # Input words.
        self.inputs = tf.compat.v1.placeholder(tf.string, shape=[1, 1], name="inputs")
        embeddings = self.dendrite.spike(self.inputs)
        self.queue = tf.queue.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=[1, 1])
        self.enqueue_op = self.queue.enqueue(embeddings)

    def _build_training_graph(self):

        self.global_step = tf.compat.v1.train.create_global_step()
        self.embeddings = self.dendrite_queue.dequeue()

        w1 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([EMBEDDING_SIZE]))
        h1 = tf.sigmoid(tf.matmul(self.embeddings, w1) + b1)

        w2 = tf.Variable(tf.random.uniform([EMBEDDING_SIZE, 1], -1.0, 1.0))
        b2 = tf.Variable(tf.zeros([1]))
        y = tf.sigmoid(tf.matmul(h1, w1) + b1)

        self.label = tf.compat.v1.placeholder(tf.float32, shape=[1], name="label")
        self.loss = tf.losses.log_loss(label, y)
        self.optimizer = tf.compat.v1.train.AdagradOptimizer(1.0).minimize(self.loss, global_step=self.global_step)

    def _preprocess_loop(self):
        try:
            with self.coord.stop_on_exception():
                while not self.coord.should_stop() and self.running:
                    next_example = next(self.cola_generator))
                    run_output = self.session.run(  fetches=[self.enqueue_op],
                                                    feed_dict={self.inputs: next_example['inputs']})
        except Exception as e:
            self.coord.request_stop(e)

    def _training_loop(self):def _train_loop(self):
        try:
            with self.coord.stop_on_exception():
                while not self.coord.should_stop() and self.running:
                    run_output = self.session.run(  fetches=self.get_fetches(),
                                                    feed_dict=self.get_feeds())
        except Exception as e:
            self.coord.request_stop(e)
        logger.info('Stopped training thread.')


    def start(self):
        self.running = True
        self.main_thread.start()

    def stop(self):
        logger.debug('Request Nucleus stop.')
        self.running = False
        if self.coord:
            self.coord.request_stop()
        self.main_thread.join()

    def _main(self):
        logger.debug('Started Nucleus training.')
        with self.session:
            try:
                # Set up threading coordinator.
                self.coord = tf.train.Coordinator()

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


def download_and_extract_cola():
    logger.info("Downloading and extracting CoLA into neurons/CoLA/data")
    data_file = "CoLA.zip"
    urllib.request.urlretrieve(CoLA_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall("neurons/CoLA/data")
    os.remove(data_file)
    logger.info("\tCompleted!")

def main():

    # Download the data.
    download_and_extract_cola()

    config = Config()

    dendrite = Dendrite()

    neuron = Neuron(config, dendrite)

    neuron.train()

    def tear_down(_config, _dendrite, _neuron):
        logger.debug('tear down.')
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
        tear_down(config, neuron)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: '+ str(e))
        tear_down(config, neuron)

if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
