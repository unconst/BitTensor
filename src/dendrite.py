import logging
import grpc
import numpy as np
import proto.bolt_pb2
import proto.bolt_pb2_grpc
import tensorflow as tf

class Dendrite():
    def __init__(self, ip_address):
        self.channel = grpc.insecure_channel(ip_address) 

    def spike(self, words_tensor):
        return tf.py_func(self.query, [words_tensor], tf.float32)

    def query(self, words):
        return np.zeros((128, 128), dtype=np.float32)
        # try:
        #     stub = proto.bolt_pb2_grpc.BoltStub(self.channel)
        #     words_proto = tf.make_tensor_proto(words)
        #     response = stub.Spike(words_proto)
        #     return response
        # except:
        #     return np.zeros(words.size, 128)