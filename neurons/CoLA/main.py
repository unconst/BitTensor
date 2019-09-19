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

class Neuron():

    def __init__(self, config, dendrite):
        self.config = config
        self.dendrite = dendrite
        self.cola = cola.Cola()
        self.cola_generator = self.cola.example_generator('neurons/CoLA/data/CoLA/dev.tsv')

    def train(self):
        for _ in range(100):
            logger.info(next(self.cola_generator))


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
