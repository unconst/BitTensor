from config import Config

from metagraph import Metagraph
from dendrite import Dendrite
from nucleus import Nucleus
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

def set_timed_loops(tl, metagraph, nucleus, synapse, dendrite):

    # Pull the updated graph state (Vertices, Edges, Weights)
    @tl.job(interval=timedelta(seconds=17))
    def pull_metagraph():
        metagraph.pull_metagraph()

    # Publish attributions (Edges, Weights.)
    @tl.job(interval=timedelta(seconds=5))
    def pull_metagraph():
        metagraph.publish_attributions()

    # Load an updated inference nn-tensorflow model.
    @tl.job(interval=timedelta(seconds=15))
    def load_graph():
        synapse.load_graph()

    # Reselect downstream nodes.
    # TODO(const) perhaps this should be removed. Instead downstream state is
    # fixed at the start.
    @tl.job(interval=timedelta(seconds=13))
    def reselect_channels():
        dendrite.reselect_channels()

def serve():
    config = Config()
    logger.debug(config)

    # The metagrpah manages the global network state.
    metagraph = Metagraph(config)

    # The dendrite manages our connections to 'upstream' nodes.
    dendrite = Dendrite(config, metagraph)

    # The nucleus trains the NN object.
    nucleus = Nucleus(config, metagraph, dendrite)

    # Start the soma.
    nucleus.start()
    time.sleep(3)

    # The synapse manages our connection to downstream nodes.
    # TODO(const) Market driven bidding for neighbors with FAN-IN K value.
    synapse = BoltServicer(config, metagraph)

    # Start timed calls.
    tl = Timeloop()
    set_timed_loops(tl, metagraph, nucleus, synapse, dendrite)
    tl.start(block=False)

    # Serve the synapse on a grpc server.
    server_address = config.address + ":" + config.port
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(synapse, grpc_server)
    grpc_server.add_insecure_port(server_address)
    logger.debug('served synapse on: {}.', server_address)
    grpc_server.start()

    def tear_down():
        grpc_server.stop(0)
        nucleus.stop()
        del metagraph
        del dendrite
        del nucleus
        del synapse

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)

    except KeyboardInterrupt:
        logger.debug('keyboard interrupt.')
        tear_down()

    except:
        logger.error('unknown interrupt.')
        tear_down()


if __name__ == '__main__':
    logger.debug("started neuron.")
    serve()
