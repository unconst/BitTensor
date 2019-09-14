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
            self._graderpc(channel, message_id, grad)
        return

    def _spikerpc(self, channel, message_id, spikes):
        if channel is None:
            return np.zeros((len(spikes), EMBEDDING_SIZE))
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)
            request =  bittensor.proto.bolt_pb2.SpikeRequest(
                        sender_identity = self.config.identity,
                        message_identity = message_id,
                        payload = pickle.dumps(spikes.numpy(),  protocol=0))
            response = stub.Spike(request)
            np_response = pickle.loads(response.payload).reshape(EMBEDDING_SIZE, -1)
            return np_response

        except Exception as error:
            #logger.info('failed call {}', error)
            return np.zeros((len(spikes), EMBEDDING_SIZE))

    def _gradrpc(self, channel, message_id, grad):
        if channel is None:
            return
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)
            request = bittensor.proto.bolt_pb2.GradeRequest(
                        sender_identity = self.config.identity,
                        message_identity = message_id,
                        grad_payload = pickle.dumps(grad.numpy(),  protocol=0))
            stub.Grade(request)
        except Exception as error:
            return
