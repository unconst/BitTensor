import bittensor

import numpy as np
import grpc


class Dendrite:

    def __init__(self, config, metagraph):
        pass

    #     self.config = config
    #     self.metagraph = metagraph
    #     self.channels = [None for _ in range(self.config.k)]
    #     self.channel_ids = [None for _ in range(self.config.k)]
    #     self.connect()
    #
    # def connect(self):
    #     for i in range(self.config.k):
    #         if self.channels[i] == None:
    #             self._set_channel(i)
    #
    # def _set_channel(self, i):
    #     for node in self.metagraph.nodes.values():
    #         if node.identity in self.channel_ids:
    #             continue
    #         if node.identity == self.config.identity:
    #             continue
    #         else:
    #             address = node.address + ':' + node.port
    #             self.channels[i] = grpc.insecure_channel(address)
    #             self.channel_ids[i] = node.identity
    #             break
    #
    # def spike(self, message_id, spikes):
    #     dspikes = []
    #     for channel in self.channels:
    #         dspikes.append(self._spikerpc(channel, message_id, spikes))
    #     return dspikes
    #
    # def grade(self, message_id, dgrades):
    #     for channel, grad in zip(self.channels, dgrades):
    #         self._gradrpc(channel, message_id, grad)
    #     return
    #
    # def _spikerpc(self, channel, message_id, spikes):
    #
    #     # If channel is empty. Return Zeros.
    #     if channel is None:
    #         return np.zeros((len(spikes), 128))
    #
    #     try:
    #         # Build Stub and request proto.
    #         stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)
    #
    #         # Build payload.
    #         # TODO(const) This is a waste, deserialize to serialize again.
    #         spikes_bytes = pickle.dumps(spikes, protocol=0)
    #
    #         # Create spike request proto.
    #         request = bittensor.proto.bolt_pb2.SpikeRequest(
    #             parent_id=self.config.identity,
    #             message_id=message_id,
    #             payload=spikes_bytes)
    #
    #         # Send TCP spike request.
    #         response = stub.Spike(request)
    #
    #         # Deserialize response.
    #         return pickle.loads(response.payload).reshape(128, -1)
    #
    #     except Exception as error:
    #         #logger.info('failed call {}', error)
    #         return np.zeros((len(spikes), 128))
    #
    # def _gradrpc(self, channel, message_id, grad):
    #
    #     # If channel is empty return
    #     if channel is None:
    #         return
    #
    #     try:
    #         # Build Stub and request proto.
    #         stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)
    #
    #         # Build Grade Request proto.
    #         request = bittensor.proto.bittensor_pb2.GradeRequest(
    #             parent_id=self.config.identity,
    #             message_id=message_id,
    #             grad_payload=pickle.dumps(grad, protocol=0))
    #
    #         # Send grade request.
    #         stub.Grade(request)
    #
    #         # Pass.
    #
    #     except Exception as error:
    #         return
