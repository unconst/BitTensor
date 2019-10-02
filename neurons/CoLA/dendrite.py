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

    def grad(self, nounce, spikes, grads):
        # Type checks.
        assert(type(nounce) == str)
        assert(type(spikes) == type(np.array([])))
        assert(type(grads) == list)
        assert(type(grads[0]) == type(np.array([])))

        # Encode nounce and source.
        nounce_bytes = bytes(nounce, 'utf-8')
        source_bytes = bytes(self.config.identity, 'utf-8')
        spikes_bytes = pickle.dumps(spikes, protocol=0)

        # Create message hash.
        hash = SHA256.new()
        hash.update(nounce_bytes)
        hash.update(source_bytes)
        hash.update(spikes_bytes)
        message_hash = hash.digest()

        #logger.info('nounce {} hash {}', nounce, message_hash)

        # Query downstream.
        for (i, channel) in enumerate(self.channels):
            # Check channel exists.
            if not channel:
                continue

            # Encode gradient for this channel.
            grad_bytes = pickle.dumps(grads[i], protocol=0)

            # Create request proto.
            request = bittensor.proto.bittensor_pb2.GradeRequest(
                version=1.0,
                source_id=self.config.identity,
                parent_id=self.config.identity,
                message_id=message_hash,
                payload=grad_bytes)

            try:
                # Build stub.
                stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

                # Send non-waiting Grade request.
                stub.Grade.future(request)
            except:
                pass

        return

    def spike(self, nounce, spikes):
        # Type checks.
        assert(type(nounce) == str)
        assert(type(spikes) == type(np.array([])))

        # Encode nounce and source.
        nounce_bytes = bytes(nounce, 'utf-8')
        source_bytes = bytes(self.config.identity, 'utf-8')
        payload_bytes = pickle.dumps(spikes, protocol=0)

        # Create message hash.
        hash = SHA256.new()
        hash.update(nounce_bytes)
        hash.update(source_bytes)
        hash.update(payload_bytes)
        message_hash = hash.digest()

        #logger.info('nounce {} hash {}', nounce, message_hash)

        # Build request proto.
        request = bittensor.proto.bittensor_pb2.SpikeRequest(
            version=1.0,
            source_id=self.config.identity,
            parent_id=self.config.identity,
            message_id=message_hash,
            payload=payload_bytes)

        # Query downstream.
        futures = []
        for channel in self.channels:
            # Check channel exists.
            if channel == None:
                futures.append(None)
                continue

            try:
                # Build channel
                # TODO(const): having prebuilt stubs would be better.
                stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

                # Send non-waiting spike request.
                futures.append(stub.Spike.future(request, timeout=1))
            except:
                pass

        return futures


    def _fill_futures_or_none(self, futures):
        # Build result buffer.
        result = []
        for _ in futures:
            zeros = np.zeros((self._batch_size, EMBEDDING_SIZE))
            result.append(zeros)

        # Fill futures or ttl.
        while True:
            remaining = len(futures)
            for i, future in enumerate(futures):
                if future == None:
                    remaining -= 1
                elif future.done():
                    remaining -= 1
                    try:
                        response = future.result()
                        dspikes = pickle.loads(response.payload)
                        result[i] = dspikes.reshape(-1, EMBEDDING_SIZE)
                    except Exception as e:
                        pass
            if remaining == 0:
                break

        return result
