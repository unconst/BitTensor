from config import Config
from dendrite import Dendrite
from metagraph import Metagraph
from neuron import Neuron
from synapse import BoltServicer

from concurrent import futures
from loguru import logger
import sys
import time

import grpc
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

def serve():

    config = Config()
    logger.info("Config: {}", config)

    # The metagrpah manages the global network state.
    # TODO(const) Make this not a stub.
    metagraph = Metagraph(config)

    # The dendrite manages our connections to downstream nodes.
    dendrite = Dendrite(config, metagraph)

    # The neuron manages our internal learner.
    neuron = Neuron(config, dendrite)
    neuron.start()
    time.sleep(3)

    # The synapse manages our connection to upstream nodes.
    synapse = BoltServicer(config)

    # Serve the synapse.

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(synapse, grpc_server)
    grpc_server.add_insecure_port(config.address)
    grpc_server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpc_server.stop(0)
        neuron.stop()

if __name__ == '__main__':
    logger.info("BitTensor.")
    serve()
