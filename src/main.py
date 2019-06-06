from config import Config
from dendrite import Dendrite
from metagraph import Metagraph
from neuron import Neuron
from synapse import BoltServicer

from concurrent import futures
from loguru import logger
import sys
import time
from timeloop import Timeloop
from datetime import timedelta


import grpc
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

def set_timed_loops(tl, metagraph, neuron, synapse, dendrite):

    @tl.job(interval=timedelta(seconds=10))
    def pull_metagraph():
        metagraph.pull_metagraph()

    @tl.job(interval=timedelta(seconds=15))
    def load_graph():
        synapse.load_graph()

    @tl.job(interval=timedelta(seconds=10))
    def reselect_channels():
        dendrite.reselect_channels()

def serve():
    config = Config()
    logger.info("Config: {}", config)

    # The metagrpah manages the global network state.
    metagraph = Metagraph(config)

    # The dendrite manages our connections to 'upstream' nodes.
    dendrite = Dendrite(config, metagraph)

    # The neuron manages our internal learner.
    neuron = Neuron(config, dendrite)
    neuron.start()
    time.sleep(3)

    # The synapse manages our connection to 'downstream' nodes.
    synapse = BoltServicer(config)

    # Start timed calls on our Neuron.
    tl = Timeloop()
    set_timed_loops(tl, metagraph, neuron, synapse, dendrite)
    tl.start(block=True)

    # Serve the synapse.
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(synapse, grpc_server)
    grpc_server.add_insecure_port(config.address)
    grpc_server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('stop')
        grpc_server.stop(0)
        neuron.stop()
        del metagraph
        del dendrite
        del synapse
        del neuron

if __name__ == '__main__':
    logger.info("BitTensor.")
    serve()
