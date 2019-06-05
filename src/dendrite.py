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
    def __init__(self, config, metagraph):
        self.config = config
        self.channels = [None for _ in range(self.config.k)]

        # self.ip_addresses = metagraph.remote_neurons
        # for addr in self.ip_addresses:
        #     self.channels.append(grpc.insecure_channel(addr))

    def spike(self, is_training, words_tensor, embedding_dim):
        # TODO(const) Implement distillation here for inference.
        # TODO(const) Implement sub networks for each dendrite.
        return_dtypes = [tf.float32 for _ in range(self.config.k)]
        function_input = [is_training, words_tensor, embedding_dim]
        return tf.cond(tf.equal(is_training, tf.constant(True)),
                    true_fn=lambda: tf.py_function(func=self.query, inp=function_input, Tout=return_dtypes),
                    false_fn=lambda: tf.py_function(func=self.query, inp=function_input, Tout=return_dtypes))


    def query(self, is_training, words, embedding_dim):
        # TODO(const) Currently this function is syncronous. Calls to the
        # dendrite nodes should be async to save on time.

        result = []
        for i in range(self.config.k):
            result.append(np.zeros((len(words), embedding_dim), dtype=np.float32))

        if is_training:
            for i in range(self.config.k):
                channel = self.channels[i]
                if channel:
                    result[i] = self._send_spike(channel, words)

        return result


    def _send_spike(channel, words):
        # Build Stub and send spike.
        stub = proto.bolt_pb2_grpc.BoltStub(channel)
        words_proto = tf.make_tensor_proto(words)
        response = stub.Spike(words_proto)

        # Deserialize response.
        # TODO(const) This should be a special tf.operation.
        response_shape = [dim.size for dim in response.tensor_shape.dim]
        assert(response_shape[1] == embedding_dim)
        np_response = _bytes_to_np(response.tensor_content, response_shape)
        return np_response
