import neuron
import dendrite
import synapse

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
    identity = my_ip.split(']:')[1]

    dendrite_ips = []
    for addr in sys.argv[2:]:
        dendrite_ips.append(addr)

    logger.info("Node IP: {}", my_ip)
    logger.info("Node ID: {}", identity)
    logger.info("Dendrite IPs: {}", dendrite_ips)

    dend = dendrite.Dendrite(dendrite_ips)

    nn = neuron.Neuron(identity, dend)
    nn.start()
    time.sleep(3)

    bolt = synapse.BoltServicer(identity)
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
