import bittensor

from Crypto.Hash import SHA256
import grpc
from loguru import logger
import numpy as np
import pickle
import tensorflow as tf

EMBEDDING_SIZE=128

class Dendrite():

    def __init__(self, config, metagraph):
        self.config = config
        self.metagraph = metagraph
        self.channels = [None for _ in range(self.config.k)]
        self.channel_nodes = [None for _ in range(self.config.k)]
        self.select_channels()

    def select_channels(self):
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


    def spike (self, text_tensor):
        rtypes = [tf.float32 for _ in range(self.config.k)]
        inputs = [text_tensor]
        return tf.py_function(self._spike, inputs, rtypes)

    def _spike(self, spikes):
        result = []
        for i in range(self.config.k):
            res = self._spikerpc(self.channels[i], spikes)
            if res is None:
                result.append(np.zeros( (self.config.batch_size, EMBEDDING_SIZE), dtype=np.float32))
            else:
                result.append(res)
        return result

    def _spikerpc(self, channel, spikes):
        if channel is None:
            return None

        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Build message hash
            identity_bytes = self.config.identity.encode()
            spike_bytes = pickle.dumps(spikes.numpy(),  protocol=0)

            # Create hash from self.identity and spikes.
            hash = SHA256.new()
            hash.update(identity_bytes)
            hash.update(spike_bytes)
            message_hash = hash.digest()

            # Build request proto.
            request = bittensor.proto.bittensor_pb2.SpikeRequest(
                        parent_id = self.config.identity,
                        message_id = message_hash,
                        payload = spike_bytes)

            # Send spike request.
            response = stub.Spike(request)

            # Deserialize response as numpy.
            return pickle.loads(response.payload).reshape(-1, EMBEDDING_SIZE)

        except Exception as error:
            logger.info('failed call {}', error)
            return None

    def grade (self, spikes, grads):
        inputs = [spikes] + grads
        return tf.py_function(self._grad, inputs, [])

    def _grad(self, spikes, *grads):
        for i in range(self.config.k):
            channel = self.channels[i]
            grad_i = grads[i]
            if channel:
                self._gradrpc(channel, spikes, grad_i)

    def _gradrpc(self, channel, spikes, grad):
        if channel is None:
            return

        try:
            # Build Stub.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Build message hash
            identity_bytes = self.config.identity.encode()
            grad_bytes = pickle.dumps(grad.numpy(),  protocol=0)
            spike_bytes = pickle.dumps(spikes.numpy(),  protocol=0)

            # Create hash from self.id and spikes.
            hash = SHA256.new()
            hash.update(identity_bytes)
            hash.update(spike_bytes)
            message_hash = hash.digest()

            # Create request proto.
            request = bittensor.proto.bittensor_pb2.GradeRequest(
                        parent_id = self.config.identity,
                        message_id = message_hash,
                        payload = grad_bytes)

            # Send Grade request.
            stub.Grade(request)

            # Pass.

        except Exception as error:
            #logger.info('failed call {}', error)
            pass
