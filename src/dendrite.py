import logging
import grpc
import proto.bolt_pb2
import proto.bolt_pb2_grpc
import numpy as np
import struct
import tensorflow as tf

def _bytes_to_np(in_bytes, shape):
    length = len(in_bytes)/4
    data = struct.unpack('<%df' % length, in_bytes)
    data_array = np.array(data, np.float32)
    out = np.reshape(data_array, shape)
    return out

class Dendrite():
    def __init__(self, ip_address):
        self.channel = grpc.insecure_channel(ip_address)

    def spike(self, is_training, words_tensor, embedding_dim):
        return tf.cond(tf.equal(is_training, tf.constant(True)),
                    true_fn=lambda: self._run_spike(words_tensor, embedding_dim),
                    false_fn=lambda: self._zeros(words_tensor, embedding_dim))

    def _run_spike(self, words_tensor, embedding_dim):
        return tf.py_func(self.query, [words_tensor, embedding_dim], tf.float32)

    def _zeros(self, words_tensor, embedding_dim):
        return tf.zeros([tf.shape(words_tensor)[0], embedding_dim])

    def query(self, words, embedding_dim):
        batch_size = len(words)

        try:
            stub = proto.bolt_pb2_grpc.BoltStub(self.channel)
            words_proto = tf.make_tensor_proto(words)
            response = stub.Spike(words_proto, timeout=0.01)
            response_shape = [dim.size for dim in response.tensor_shape.dim]
            assert(response_shape[1] == embedding_dim)
            np_out = _bytes_to_np(response.tensor_content, response_shape)
            return np_out

        except Exception as e:
            return np.zeros((batch_size, embedding_dim), dtype=np.float32)
