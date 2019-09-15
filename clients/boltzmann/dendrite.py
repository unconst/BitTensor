import bittensor

import numpy as np

class Dendrite:

    def __init__(self, config):
        self.config = config
        self.channels = [None for _ in range(self.config.k)]

    def spike(self, message_id, spikes):
        dspikes = []
        for channel in self.channels:
            dspikes.append(self._spikerpc(channel, message_id, spikes))
        return dspikes

    def grade(self, message_id, dgrades):
        for channel, grad in zip(self.channels, dgrades):
            self._gradrpc(channel, message_id, grad)
        return

    def _spikerpc(self, channel, message_id, spikes):

        # If channel is empty. Return Zeros.
        if channel is None:
            return np.zeros((len(spikes), 128))

        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)

            # Build payload.
            # TODO(const) This is a waste, deserialize to serialize again.
            spikes_bytes = pickle.dumps(spikes,  protocol=0)

            # Create spike request proto.
            request =  bittensor.proto.bolt_pb2.SpikeRequest(
                        sender_identity = self.config.identity,
                        message_identity = message_id,
                        payload = spikes_bytes)

            # Send TCP spike request.
            response = stub.Spike(request)

            # Deserialize response.
            return pickle.loads(response.payload).reshape(128, -1)

        except Exception as error:
            #logger.info('failed call {}', error)
            return np.zeros((len(spikes), 128))

    def _gradrpc(self, channel, message_id, grad):

        # If channel is empty return
        if channel is None:
            return

        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)

            # Build Grade Request proto.
            request = bittensor.proto.bolt_pb2.GradeRequest(
                        sender_identity = self.config.identity,
                        message_identity = message_id,
                        grad_payload = pickle.dumps(grad,  protocol=0))

            # Send grade request.
            stub.Grade(request)

            # Pass.

        except Exception as error:
            return
