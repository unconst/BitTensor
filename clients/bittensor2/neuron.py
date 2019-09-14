import bittensor

from concurrent import futures
import grpc
from loguru import logger

class Buffer:
    def __init__(self,  sender_id = None,
                        message_id = None,
                        lspikes = None,
                        rspikes = None,
                        dspikes = None,
                        lgrades = None):

        self.sender_id = sender_id
        self.message_id = message_id
        self.lspikes = lspikes
        self.rspikes = rspikes
        self.dspikes = dspikes
        self.lgrades = lgrades

    def set(self, sender_id = None,
                  message_id = None,
                  lspikes = None,
                  rspikes = None,
                  dspikes = None,
                  lgrades = None ):

        if not self.sender_id:
            self.sender_id = sender_id
        if not self.message_id:
            self.message_id = message_id
        if not self.lspikes:
            self.lspikes = lspikes
        if not self.rspikes:
            self.rspikes = rspikes
        if not self.dspikes:
            self.dspikes = dspikes
        if not self.lgrades:
            self.lgrades = lgrades


class Neuron(bittensor.proto.bolt_pb2_grpc.BoltServicer):

    def __init__(self, config, dendrite, nucleus, metagraph):
        self.config = config
        self.dendrite = dendrite
        self.nucleus = nucleus
        self.metagraph = metagraph
        self.memory = {}

        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor.proto.bolt_pb2_grpc.add_BoltServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.server_address)

    def __del__(self):
        self.server.stop(0)
        logger.debug('Stopped Serving Neuron at: {}.', self.server_address)

    def serve(self):
        self.server.start()
        logger.debug('Started Serving Neuron at: {}.', self.server_address)

    def Spike(self, request, context):

        # Unpack message.
        sender_id = request.sender_identity
        message_id = request.message_identity
        uspikes = pickle.loads(request.payload)

        # Check for repsonse in buffer.
        if message_id in self.memory:
            # Return local spikes.
            return self.memory[message_id].lspikes

        # Get downstream spikes.
        dspikes = self.dendrite.spike(message_id, uspikes)

        # Get local spikes.
        lspikes = self.nucleus.spike(uspikes, dspikes)

        # Save to buffer.
        self.memory[message_id] = Buffer(message_id = message_id,
                                      sender_id = sender_id,
                                      lspikes = lspikes,
                                      rspikes = rspikes,
                                      dspikes = dspikes)

        # Pack response.
        payload = pickle.dumps(lspikes, protocol=0)
        response = bittensor.proto.bolt_pb2.SpikeResponse(
                        responder_identity = self.config.identity,
                        message_identity = message_id,
                        payload = payload)

        return response


    def Grade(self, request, context):

        # Unpack request.
        sender_id = request.sender_identity
        message_id = request.message_identity
        ugrades = pickle.loads(request.payload)

        # Check for lost or badly routed grades.
        if message_id not in self.memory:
            return

        # Get local spikes.
        mem_buffer = self.memory[message_id]
        lspikes = mem_buffer.lspikes

        # Get downstream spikes.
        dspikes = mem_buffer.dspikes

        # Get upstream spikes
        uspikes = mem_buffer.dspikes

        # Get downstream grads and local grads.
        dgrades, lgrades = self.nucleus.grade(ugrades, uspikes, dspikes)

        # Set lgrades in buffer.
        mem_buffer.lgrades = lgrades

        # Send downstream grads.
        self.dendrite.grade(message_id, dgrades)

        return bittensor.proto.bolt_pb2.GradeResponse(accept=True)

    def Train (self):
        logger.debug('train step.')
        pass
        ## Get the next batch of gradients
        #batch = self.memory.gradient_batch()

        # Apply them to Nucleus
        #self.nucleus.train(batch)
