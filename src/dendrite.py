import grpc
from loguru import logger
import proto.bolt_pb2
import proto.bolt_pb2_grpc
import numpy as np
import struct
import tensorflow as tf
import time

def _bytes_to_np(in_bytes, shape):
    length = len(in_bytes)/4
    data = struct.unpack('<%df' % length, in_bytes)
    data_array = np.array(data, np.float32)
    out = np.reshape(data_array, shape)
    return out


# TODO (const): Negotiate channels with upstream nodes.

class Dendrite():
    def __init__(self, config, metagraph):
        self.config = config
        self.metagraph = metagraph
        self.channels = [None for _ in range(self.config.k)]
        self.channel_nodes = [None for _ in range(self.config.k)]
        self.reselect_channels()

    def reselect_channels(self):
        nodes = self.metagraph.nodes
        for i in range(self.config.k):
            if self.channels[i] != None:
                continue

            selected_node = None
            for node in nodes.values():
                if node not in self.channel_nodes and node.identity != self.config.identity:
                    selected_node = node
                    break

            if selected_node:
                address = selected_node.address + ':' + selected_node.port
                self.channels[i] = grpc.insecure_channel(address)
                self.channel_nodes[i] = selected_node

        logger.debug(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_rep = "\ndendrite = {\n"
        i = 0
        for node in self.channel_nodes:
            if node:
                str_rep += ('\t\t[' + str(i) + ']:' + str(node.identity) + "\n")
            else:
                str_rep += ('\t\t[' + str(i) + ']:' + "None" + "\n")
            i+=1
        str_rep += "}."
        return  str_rep



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
                    res = self._send_spike(channel, words, embedding_dim)
                    if res is not None:
                        result[i] = res

        return result


    def _send_spike(self, channel, words, embedding_dim):
        try:
            # Build Stub and send spike.
            stub = proto.bolt_pb2_grpc.BoltStub(channel)
            words_proto = tf.make_tensor_proto(words)
            response = stub.Spike(words_proto)

            # Deserialize response.
            # TODO(const) This should be a special tf.operation.
            response_shape = [dim.size for dim in response.tensor_shape.dim]
            np_response = _bytes_to_np(response.tensor_content, response_shape)
            return np_response

        except Exception as error:
            #logger.info('failed call {}', error)
            #time.sleep(10)
            #logger.info('.')
            return None
