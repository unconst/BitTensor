from config import Config

from metagraph import Metagraph
from dendrite import Dendrite
from nucleus import Nucleus
from synapse import BoltServicer
import visualization

from concurrent import futures
from loguru import logger
import sys
import time
from timeloop import Timeloop
from datetime import timedelta

import grpc
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*(23.93446989)

def set_timed_loops(tl, metagraph, nucleus, synapse, dendrite):

    # Pull the updated graph state (Vertices, Edges, Weights)
    @tl.job(interval=timedelta(seconds=7))
    def pull_metagraph():
        metagraph.pull_metagraph()

    # Publish attributions (Edges, Weights.)
    @tl.job(interval=timedelta(seconds=2))
    def publish_attributions():
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

    # The synapse manages our connection to downstream nodes.
    # TODO(const) Market driven bidding for neighbors with FAN-IN K value.
    synapse = BoltServicer(config, metagraph)
    logger.info('Started Synapse.')

    # Start the Nucleus.
    nucleus.start()
    logger.info('Started Nucleus.')

    # Start timed calls.
    tl = Timeloop()
    set_timed_loops(tl, metagraph, nucleus, synapse, dendrite)
    tl.start(block=False)
    logger.info('Started Timers.')

    # Serve the synapse on a grpc server.
    server_address = config.address + ":" + config.port
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(synapse, grpc_server)
    grpc_server.add_insecure_port(server_address)
    logger.debug('Served synapse on: {}.', server_address)
    grpc_server.start()

    def tear_down(_server, _nucleus, _metagraph, _dendrite, _synapse):
        _server.stop(0)
        _nucleus.stop()
        del _metagraph
        del _dendrite
        del _nucleus
        del _synapse

    try:
        logger.info('Begin wait on Main.')
        while True:
            image_buffer = visualization.generate_edge_weight_buffer(metagraph.nodes)
            nucleus.update_metagraph_summary(image_buffer)
            logger.info('Updated metagraph image.')
            time.sleep(5)

    except KeyboardInterrupt:
        logger.debug('keyboard interrupt.')
        tear_down(grpc_server, nucleus, metagraph, dendrite, synapse)

    except:
        logger.error('unknown interrupt.')
        tear_down(grpc_server, nucleus, metagraph, dendrite, synapse)


if __name__ == '__main__':
    logger.debug("started neuron.")
    serve()
