import neuron
import dendrite
import synapse
import metagraph

from concurrent import futures
from loguru import logger
import sys
import time

import grpc
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

def serve():
    # The metagrpah manages the global network state.
    # TODO(const) Make this not a stub.
    this_metagraph = metagraph.Metagraph(sys.argv)
    logger.info("Node IP: {}", this_metagraph.this_address)
    logger.info("Node ID: {}", this_metagraph.this_identity)
    logger.info("Dendrite IPs: {}", this_metagraph.remote_neurons)

    # The dendrite manages our connections to downstream nodes.
    this_dendrite = dendrite.Dendrite(this_metagraph)

    # The neuron manages our internal learner.
    this_neuron = neuron.Neuron(this_metagraph, this_dendrite)
    this_neuron.start()
    time.sleep(3)

    # The synapse manages our connection to upstream nodes.
    this_synapse = synapse.BoltServicer(this_metagraph)

    # Serve the synapse.
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(this_synapse, grpc_server)
    grpc_server.add_insecure_port(this_metagraph.this_address)
    grpc_server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpc_server.stop(0)
        this_neuron.stop()

if __name__ == '__main__':
    logger.info("BitTensor.")
    serve()
