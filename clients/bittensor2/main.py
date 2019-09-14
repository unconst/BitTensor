import bittensor

from config import Config
from metagraph import Metagraph
from dendrite import Dendrite
from nucleus import Nucleus
from neuron import Neuron

from loguru import logger
import time
from timeloop import Timeloop
from datetime import timedelta

def set_timed_loops(tl, neuron):

    # Apply a gradient step.
    @tl.job(interval=timedelta(seconds=3))
    def train():
        neuron.Train()

def main():

    config = Config()

    metagraph = Metagraph(config)

    dendrite = Dendrite(config)

    nucleus = Nucleus(config)

    neuron = Neuron(config, dendrite, nucleus, metagraph)

    neuron.serve()

    # Start timed calls.
    tl = Timeloop()
    set_timed_loops(tl, neuron)
    tl.start(block=False)
    logger.info('Started Timers.')

    def tear_down(_config, _neuron, _dendrite, _nucleus, _metagraph):
        logger.debug('tear down.')
        del _neuron
        del _dendrite
        del _nucleus
        del _metagraph
        del _config

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(config, neuron, dendrite, nucleus, metagraph)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: '+ str(e))
        tear_down(config, neuron, dendrite, nucleus, metagraph)

if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
