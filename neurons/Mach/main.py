import bittensor

from config import Config
from metagraph import Metagraph
from nucleus import Nucleus
from neuron import Neuron

import argparse
from datetime import timedelta
import grpc
from loguru import logger
import random
import time
from timeloop import Timeloop


def set_timed_loops(tl, config, neuron, metagraph):

    # Pull the updated graph state (Vertices, Edges, Weights)
    @tl.job(interval=timedelta(seconds=7))
    def pull_metagraph():
        metagraph.pull_metagraph()

    # Reselect channels.
    @tl.job(interval=timedelta(seconds=10))
    def connect():
        neuron.connect()

    # Apply a gradient step.
    @tl.job(interval=timedelta(seconds=3))
    def learn():
        neuron.Learn()


def main(hparams):

    metagraph = Metagraph(hparams)

    nucleus = Nucleus(hparams)

    neuron = Neuron(hparams, nucleus, metagraph)

    neuron.serve()

    # Start timed calls.
    tl = Timeloop()
    set_timed_loops(tl, hparams, neuron, metagraph)
    tl.start(block=False)
    logger.info('Started Timers.')

    def tear_down(_hparams, _neuron, _dendrite, _nucleus, _metagraph):
        logger.debug('tear down.')
        del _neuron
        del _dendrite
        del _nucleus
        del _metagraph
        del _hparams

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(hparams, neuron, dendrite, nucleus, metagraph)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: ' + str(e))
        tear_down(hparams, neuron, dendrite, nucleus, metagraph)


if __name__ == '__main__':
    logger.debug("started neuron.")
    parser = argparse.ArgumentParser()

    # Server parameters.
    parser.add_argument(
        '--identity',
        default='abcd',
        type=str,
        help="network identity. Default identity=abcd")
    parser.add_argument(
        '--serve_address',
        default='0.0.0.0',
        type=str,
        help="Address to server neuron. Default serve_address=0.0.0.0")
    parser.add_argument(
        '--bind_address',
        default='0.0.0.0',
        type=str,
        help="Address to bind neuron. Default bind_address=0.0.0.0")
    parser.add_argument(
        '--port',
        default='9090',
        type=str,
        help="Port to serve neuron on. Default port=9090")
    parser.add_argument(
        '--eosurl',
        default='http://0.0.0.0:8888',
        type=str,
        help="Address to eos chain. Default eosurl=http://0.0.0.0:8888")
    parser.add_argument(
        '--logdir',
        default="/tmp/",
        type=str,
        help="logging output directory. Default logdir=/tmp/")

    # Nucleus parameters.
    parser.add_argument(
        '--corpus_path',
        default='neurons/Mach/data/text8.zip',
        type=str,
        help='Path to corpus of text. Default corpus_path=neurons/Mach/data/text8.zip')
    parser.add_argument(
        '--vocabulary_size',
        default=50000,
        type=int,
        help='Size fof corpus vocabulary. Default vocabulary_size=50000')
    parser.add_argument(
        '--num_sampled',
        default=64,
        type=int,
        help='Number of negative examples to sample during training. Default num_sampled=64')
    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help='Component learning rate. Default learning_rate=1e-4')
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')
    parser.add_argument(
        '--n_components',
        default=2,
        type=int,
        help='The number of training iterations. Default n_components=2')
    parser.add_argument(
        '--n_iterations',
        default=10000,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument('--n_hidden1',
                        default=512,
                        type=int,
                        help='Size of layer 1. Default n_hidden1=512')
    parser.add_argument('--n_hidden2',
                        default=512,
                        type=int,
                        help='Size of layer 1. Default n_hidden2=512')
    parser.add_argument(
        '--n_shidden1',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 1. Default n_shidden1=512')
    parser.add_argument(
        '--n_shidden2',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 2. Default n_shidden2=512')
    parser.add_argument(
        '--use_joiner_network',
        default=False,
        type=bool,
        help=
        'Do we combine downstream spikes using a trainable network. Default use_joiner_network=False'
    )
    parser.add_argument(
        '--n_jhidden1',
        default=512,
        type=int,
        help='Size of Joiner model hidden layer 1. Default n_shidden1=512')
    parser.add_argument(
        '--n_jhidden2',
        default=512,
        type=int,
        help='Size of Joinermodel hidden layer 2. Default n_shidden2=512')

    hparams = parser.parse_args()

    main(hparams)
