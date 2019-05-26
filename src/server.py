import neuron
import dendrite

from concurrent import futures
import logging
import sys
import time
import tensorflow as tf
import zipfile

import grpc
import proto.bolt_pb2
import proto.bolt_pb2_grpc

_ONE_DAY_IN_SECONDS=60*60*24

class BoltServicer(proto.bolt_pb2_grpc.BoltServicer):
    def __init__(self, identity):
        self.identity = identity
        self.load_time = 5
        self.load_graph()

    def load_graph(self):
        print ('reload graph')
        self.since_last_load = time.time()
        self.session = tf.Session() 
        self.saver = tf.train.import_meta_graph('./checkpoints/' + self.identity + '/model.meta')
        self.saver.restore(self.session, tf.train.latest_checkpoint('./checkpoints/' + self.identity))
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def Spike(self, request, context):

        if time.time() - self.since_last_load > self.load_time:
            self.load_graph()

        batch_words = [[word] for word in request.string_val]
        embeddings = self.session.run("embedding_output:0", feed_dict={"batch_words:0": batch_words, 'is_training:0': False})
        embed_proto = tf.make_tensor_proto(embeddings)
        return embed_proto


def serve():
    assert(len(sys.argv) > 2)
    my_ip = str(sys.argv[1])
    dendrite_ip = str(sys.argv[2])
    identity = my_ip.split(']:')[1]

    print ('my_ip: ' + my_ip)
    print ('dendrite_ip: ' + dendrite_ip)

    dend = dendrite.Dendrite(dendrite_ip)

    nn = neuron.Neuron(identity, dend)
    nn.start()
    time.sleep(3)

    bolt = BoltServicer(identity)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.bolt_pb2_grpc.add_BoltServicer_to_server(bolt, server)
    server.add_insecure_port(my_ip)
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
