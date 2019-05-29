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
    def __init__(self, metagraph):
        self.channels = []
        self.ip_addresses = metagraph.remote_neurons
        self.width = len(self.ip_addresses)
        for addr in self.ip_addresses:
            self.channels.append(grpc.insecure_channel(addr))

    def _run_spike(self, words_tensor, embedding_dim):
        return tf.py_func(self.query, [words_tensor, embedding_dim], [tf.float32 for _ in self.channels])

    def _zeros(self, words_tensor, embedding_dim):
        return [tf.zeros([tf.shape(words_tensor)[0], embedding_dim]) for _ in self.channels]

    def spike(self, is_training, words_tensor, embedding_dim):
        # TODO(const) Implement distillation here for inference.
        # TODO(const) Implement sub networks for each dendrite.
        return tf.cond(tf.equal(is_training, tf.constant(True)),
                    true_fn=lambda: self._run_spike(words_tensor, embedding_dim),
                    false_fn=lambda: self._zeros(words_tensor, embedding_dim))

    def query(self, words, embedding_dim):
        # TODO(const) Currently this function is syncronous. Calls to the
        # dendrite nodes should be async to save on time.
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
                # TODO(const) This should be a special tf.operation.
                response_shape = [dim.size for dim in response.tensor_shape.dim]
                assert(response_shape[1] == embedding_dim)
                np_response = _bytes_to_np(response.tensor_content, response_shape)
                result.append(np_response)

            except Exception as e:
                result.append(np.zeros((len(words), embedding_dim), dtype=np.float32))

        return result
