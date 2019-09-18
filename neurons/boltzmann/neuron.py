import bittensor

from concurrent import futures
import grpc
from loguru import logger
import pickle
import time
from threading import Lock
import queue

class Buffer:
    def __init__(self,  parent_id = None,
                        message_id = None,
                        create_time = None,
                        lspikes = None,
                        uspikes = None,
                        dspikes = None,
                        lgrads = None):

        self.parent_id = parent_id
        self.message_id = message_id
        self.create_time = create_time
        self.lspikes = lspikes
        self.uspikes = uspikes
        self.dspikes = dspikes
        self.lgrads = lgrads

    def set(self, parent_id = None,
                  message_id = None,
                  create_time = None,
                  lspikes = None,
                  uspikes = None,
                  dspikes = None,
                  lgrads = None ):

        if not self.parent_id:
            self.parent_id = parent_id
        if not self.message_id:
            self.message_id = message_id
        if not self.create_time:
            self.create_time = create_time
        if not self.lspikes:
            self.lspikes = lspikes
        if not self.uspikes:
            self.uspikes = uspikes
        if not self.dspikes:
            self.dspikes = dspikes
        if not self.lgrads:
            self.lgrads = lgrads


class Neuron(bittensor.proto.bittensor_pb2_grpc.BittensorServicer):

    def __init__(self, config, dendrite, nucleus, metagraph):
        self.config = config
        self.dendrite = dendrite
        self.nucleus = nucleus
        self.metagraph = metagraph

        self.mem_lock = Lock()
        self.memory = {}
        self.gradient_queue = queue.LifoQueue(maxsize=-1)

        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor.proto.bittensor_pb2_grpc.add_BittensorServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.server_address)

    def __del__(self):
        self.server.stop(0)
        logger.debug('Stopped Serving Neuron at: {}.', self.server_address)

    def serve(self):
        self.server.start()
        logger.debug('Started Serving Neuron at: {}.', self.server_address)

    def Spike(self, request, context):
        # Unpack message.
        parent_id = request.parent_id
        message_id = request.message_id
        uspikes = pickle.loads(request.payload)

        # Check for repsonse in buffer.
        if message_id in self.memory:
            # Return local spikes.
            lspikes = self.memory[message_id].lspikes
            payload = pickle.dumps(lspikes, protocol=0)
            response = bittensor.proto.bittensor_pb2.SpikeResponse(
                            child_id = self.config.identity,
                            message_id = message_id,
                            payload = payload)
            return response


        # Get downstream spikes.
        dspikes = self.dendrite.spike(message_id, uspikes)

        # Get local spikes.
        lspikes = self.nucleus.spike(uspikes, dspikes)

        # Save to buffer.
        self.mem_lock.acquire()
        try:
            self.memory[message_id] = Buffer(
                                          parent_id = parent_id,
                                          message_id = message_id,
                                          create_time = time.time(),
                                          lspikes = lspikes,
                                          uspikes = uspikes,
                                          dspikes = dspikes)
        finally:
            self.mem_lock.release()


        # Pack response.
        payload = pickle.dumps(lspikes, protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
                        child_id = self.config.identity,
                        message_id = message_id,
                        payload = payload)

        return response


    def Grade(self, request, context):
        # Unpack request.
        parent_id = request.parent_id
        message_id = request.message_id
        ugrades = pickle.loads(request.payload)

        # Check for lost or badly routed grades.
        if message_id not in self.memory:
            return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

        # Get local spikes.
        mem_buffer = self.memory[message_id]

        lspikes = mem_buffer.lspikes

        # Get downstream spikes.
        dspikes = mem_buffer.dspikes

        # Get upstream spikes
        uspikes = mem_buffer.uspikes

        # Get downstream grads and local grads.
        dgrades, lgrads = self.nucleus.grade(ugrades, uspikes, dspikes)

        # delete memory:
        del self.memory[message_id]

        # Put gradients on LIFO queue.
        self.gradient_queue.put(lgrads)

        # Send downstream grads.
        self.dendrite.grade(message_id, dgrades)

        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

    def Learn (self):
        # Function clears the message buffer of all outdated memory objects
        # and applies gradients from memory.
        logger.info('Learn.')

        # Clean the memory.
        self.mem_lock.acquire()
        try:
            time_now = time.time()
            to_delete = []
            for row in self.memory.values():
                if (time_now - row.create_time) > self.config.time_till_expire:
                    to_delete.append(row.message_id)

            for message_id in to_delete:
                del self.memory[message_id]

        except Exception as e:
            logger.error('Neuron failed on memory clean with Error: '+ str(e))

        finally:
            self.mem_lock.release()

        # Apply the batch.
        logger.info('Grad queue size: {}', self.gradient_queue.qsize())
        while not self.gradient_queue.empty():
            grad = self.gradient_queue.get()
            self.nucleus.learn(grad)
