import grpc
from loguru import logger
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
    def __init__(self, ip_addresses):
        self.channels = []
        self.ip_addresses = ip_addresses
        self.width = len(ip_addresses)
        for addr in ip_addresses:
            self.channels.append(grpc.insecure_channel(addr))


    def spike(self, is_training, words_tensor, embedding_dim):
        return tf.cond(tf.equal(is_training, tf.constant(True)),
                    true_fn=lambda: self._run_spike(words_tensor, embedding_dim),
                    false_fn=lambda: self._zeros(words_tensor, embedding_dim))

    def _run_spike(self, words_tensor, embedding_dim):
        return tf.py_func(self.query, [words_tensor, embedding_dim], [tf.float32 for _ in self.channels])

    def _zeros(self, words_tensor, embedding_dim):
        return [tf.zeros([tf.shape(words_tensor)[0], embedding_dim]) for _ in self.channels]

    def query(self, words, embedding_dim):
        result = []
        for i in range(self.width):
            channel = self.channels[i]
            address = self.ip_addresses[i]
            try:
                # Build Stub and send spike.
                stub = proto.bolt_pb2_grpc.BoltStub(channel)
                words_proto = tf.make_tensor_proto(words)
                response = stub.Spike(words_proto)

                # Deserialize response.
                # TODO(const) This should be a special tfoperation.
                response_shape = [dim.size for dim in response.tensor_shape.dim]
                assert(response_shape[1] == embedding_dim)
                np_response = _bytes_to_np(response.tensor_content, response_shape)
                result.append(np_response)

            except Exception as e:
                result.append(np.zeros((len(words), embedding_dim), dtype=np.float32))

        return result
