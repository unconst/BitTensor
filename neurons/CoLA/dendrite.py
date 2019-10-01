import bittensor

from Crypto.Hash import SHA256
import grpc
from loguru import logger
import numpy as np
import pickle
import tensorflow as tf

EMBEDDING_SIZE = 128


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

    def spike(self, nounce_string, text_list):

        # To be filled a list of 128 dimensional vectors.
        result = []

        # Encode nounce and source.
        nounce_bytes = bytes(nounce_string, 'utf-8')
        source_bytes = bytes(self._config.identity, 'utf-8')

        # Encode text.
        payload_byte_string = b''
        payload_bytes = []
        for string_element in text_list:
            as_bytes = bytes(string_element, 'utf-8')
            payload_bytes.append(as_bytes)
            payload_byte_string += as_bytes

        # Create message hash.
        hash = SHA256.new()
        hash.update(nounce_bytes)
        hash.update(source_bytes)
        hash.update(payload_byte_string)
        message_id = hash.digest()

        # Build request proto.
        request = bittensor.proto.bittensor_pb2.SpikeRequest(
            parent_id=self.config.identity,
            message_id=message_hash,
            payload=payload_bytes)


        # Query downstream.
        for channel in self.channels[i]:
            # Build channel
            # TODO(const): having prebuilt stubs would be better.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Send spike request.
            response = stub.Spike(request)

            response = self._spikerpc(channel, request)
            result.append()


        for i in range(self.config.k):
            res = self._spikerpc(self.channels[i], nounce_bytes, source_bytes, spikes_bytes)
            if res is None:
                result.append(
                    np.zeros((self.config.batch_size, EMBEDDING_SIZE),
                             dtype=np.float32))
            else:
                result.append(res)
        return result


    def _spikerpc(self, channel, request):

        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Send spike request.
            response = stub.Spike(request)

             response as numpy.
            return response

        except Exception as error:
            logger.info('failed call {}', error)
            return None

    def grad(self, nounce, spikes, grads):
        for i in range(self.config.k):
            channel = self.channels[i]
            grad_i = grads[i]
            if channel:
                self._gradrpc(channel, nounce, spikes, grad_i)

    def _gradrpc(self, channel, nounce, spikes, grad):
        if channel is None:
            return

        try:
            # Build Stub.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Build message hash
            nounce_bytes = pickle.dumps(nounce, protocol=0)
            identity_bytes = self.config.identity.encode()
            grad_bytes = pickle.dumps(grad, protocol=0)
            spike_bytes = pickle.dumps(spikes, protocol=0)

            # Create hash from self.id and spikes.
            hash = SHA256.new()
            hash.update(nounce_bytes)
            hash.update(identity_bytes)
            hash.update(spike_bytes)
            message_hash = hash.digest()

            logger.info('nounce {} {} spikes {} {} hash {}', nounce, nounce_bytes, spikes, spike_bytes, message_hash)

            # Create request proto.
            request = bittensor.proto.bittensor_pb2.GradeRequest(
                parent_id=self.config.identity,
                message_id=message_hash,
                payload=grad_bytes)

            # Send Grade request.
            stub.Grade(request)

            # Pass.

        except Exception as error:
            logger.info('failed call {}', error)
            pass
