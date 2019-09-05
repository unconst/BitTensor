import grpc
from loguru import logger
import pickle
import proto.bolt_pb2
import proto.bolt_pb2_grpc
import numpy as np
import random
import struct
import tensorflow as tf
from tensorflow.python.framework import ops
import time

# TODO (const): Negotiate channels with upstream nodes.

EMBEDDING_SIZE=128

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

    def _gradrpc(self, channel, spikes, grad):
        try:
            # Build Stub and request proto.
            stub = proto.bolt_pb2_grpc.BoltStub(channel)
            request = proto.bolt_pb2.GradeRequest(
                        sender_identity = self.config.identity,
                        message_identity = str(random.randint(0,1000000000)),
                        spike_payload = pickle.dumps(spikes.numpy(),  protocol=0),
                        grad_payload = pickle.dumps(grad.numpy(),  protocol=0))
            stub.Grade(request)
        except Exception as error:
            pass

    def _spikerpc(self, channel, spikes):
        #logger.info('dendrite._spikerpc')
        if channel is None:
            return None

        try:
            # Build Stub and request proto.
            stub = proto.bolt_pb2_grpc.BoltStub(channel)
            request = proto.bolt_pb2.SpikeRequest(
                        sender_identity = self.config.identity,
                        message_identity = str(random.randint(0,1000000000)),
                        payload = pickle.dumps(spikes.numpy(),  protocol=0))
            response = stub.Spike(request)
            np_response = pickle.loads(response.payload).reshape(EMBEDDING_SIZE, -1)
            return np_response

        except Exception as error:
            #logger.info('failed call {}', error)
            return None

    def _grad(self, spikes, *grads):
        for i in range(self.config.k):
            channel = self.channels[i]
            grad_i = grads[i]
            if channel:
                self._gradrpc(channel, spikes, grad_i)

    def _spike(self, spikes):
        #logger.info('dendrite._spikecast')
        # TODO(const) Currently this function is syncronous. Calls to the
        # dendrite nodes should be async to save on time.
        result = []
        for i in range(self.config.k):
            res = self._spikerpc(self.channels[i], spikes)
            if res is None:
                result.append(np.zeros((len(spikes), EMBEDDING_SIZE), dtype=np.float32))
            else:
                result.append(res)
        return result

    def grade (self, spikes, grads):
        inputs = [spikes] + grads
        return tf.py_function(self._grad, inputs, [])

    def spike (self, words_tensor):
        #logger.info('dendrite.spike')
        rtypes = [tf.float32 for _ in range(self.config.k)]
        inputs = [words_tensor]
        return tf.py_function(self._spike, inputs, rtypes)
