import neuron
import dendrite
import server

from concurrent import futures
from loguru import logger
import sys
import time

import grpc
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

def serve():
    assert(len(sys.argv) > 2)
    my_ip = str(sys.argv[1])
    dendrite_ip = str(sys.argv[2])
    identity = my_ip.split(']:')[1]

    logger.info("Node IP: {}", my_ip)
    logger.info("Node ID: {}", identity)
    logger.info("Dendrite IP: {}", dendrite_ip)

    dend = dendrite.Dendrite(dendrite_ip)

    nn = neuron.Neuron(identity, dend)
    nn.start()
    time.sleep(3)

    bolt = server.BoltServicer(identity)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(bolt, grpc_server)
    grpc_server.add_insecure_port(my_ip)
    grpc_server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpc_server.stop(0)
        nn.stop()

if __name__ == '__main__':
    logger.info("BitTensor.")
    serve()