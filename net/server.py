import neuron

from concurrent import futures
import logging
import time
import tensorflow as tf
import zipfile

import grpc
import proto.bolt_pb2
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

class BoltServicer(proto.bolt_pb2_grpc.BoltServicer):
    def __init__(self, neuron):
        self.neuron = neuron

    def Spike(self, request, context):
        embeddings = self.neuron.spike(request.words)
        embed_proto = tf.make_tensor_proto(embeddings)
        return embed_proto


def serve():

    nn = neuron.Neuron()
    nn.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(
        BoltServicer(nn), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        nn.stop()

if __name__ == '__main__':
    logging.basicConfig()
    serve()

