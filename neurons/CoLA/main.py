import bittensor

from config import Config

from loguru import logger
import urllib.request
import zipfile

# URL for downloading COLA data.
CoLA_URL ="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"

class Dendrite():

    def __init__(self):
        pass

class Neuron():

    def __init__(self):
        pass

def download_and_extract_cola():
    logger.info("Downloading and extracting CoLA into neurons/CoLA/data")
    data_file = "neurons/CoLA/data.zip"
    urllib.request.urlretrieve(CoLA_URL, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")

def main():

    # Download the data.
    download_and_extract_cola()

    config = Config()

    neuron = Neuron(config)

    neuron.serve()

    def tear_down(_config, _neuron):
        logger.debug('tear down.')
        del _neuron
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
