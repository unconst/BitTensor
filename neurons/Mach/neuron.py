import bittensor

from concurrent import futures
import grpc
from loguru import logger
import numpy as np
import pickle
import time
from threading import Lock
import queue


class Buffer:

    def __init__(self,
                 source_id=None,
                 parent_id=None,
                 message_id=None,
                 create_time=None,
                 lspikes=None,
                 uspikes=None,
                 dspikes=None,
                 lgrads=None):

        self.source_id = source_id
        self.parent_id = parent_id
        self.message_id = message_id
        self.create_time = create_time
        self.lspikes = lspikes
        self.uspikes = uspikes
        self.dspikes = dspikes
        self.lgrads = lgrads

    def set(self,
            source_id=None,
            parent_id=None,
            message_id=None,
            create_time=None,
            lspikes=None,
            uspikes=None,
            dspikes=None,
            lgrads=None):

        if not self.source_id:
            self.source_id = source_id
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

        self.lock = Lock()
        self.memory = {}
        self.gradient_queue = queue.LifoQueue(maxsize=-1)

        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor.proto.bittensor_pb2_grpc.add_BittensorServicer_to_server(
            self, self.server)
        self.server.add_insecure_port(self.server_address)

        self.channels = [None for _ in range(self.config.k)]
        self.channel_ids = [None for _ in range(self.config.k)]
        self.connect()

    def connect(self):
        for i in range(self.config.k):
            if self.channels[i] == None:
                self._set_channel(i)

    def _set_channel(self, i):
        for node in self.metagraph.nodes.values():
            if node.identity in self.channel_ids:
                continue
            if node.identity == self.config.identity:
                continue
            else:
                address = node.address + ':' + node.port
                self.channels[i] = grpc.insecure_channel(address)
                self.channel_ids[i] = node.identity
                break

    def __del__(self):
        self.server.stop(0)
        logger.debug('Stopped Serving Neuron at: {}.', self.server_address)

    def serve(self):
        self.server.start()
        logger.debug('Started Serving Neuron at: {}.', self.server_address)

    def _spike_future(self, channel, request):
        if channel == None:
            return None
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Create spike request proto.
            request = bittensor.proto.bittensor_pb2.SpikeRequest(
                version=1.0,
                source_id=request.source_id,
                parent_id=self.config.identity,
                message_id=request.message_id,
                payload=request.payload)

            # Send TCP spike request with futures callback.
            return stub.Spike.future(request)
        except:
            return None

    def _fill_dspikes(self, dspikes, futures):
        while True:
            remaining_futures = self.config.k
            for i in range(self.config.k):
                if futures[i] != None:
                    if futures[i].done():
                        remaining_futures -= 1
                        try:
                            result = futures[i].result()
                            next_dspikes = pickle.loads(result.payload).reshape(
                                -1, 128)
                            dspikes[i] = next_dspikes
                        except:
                            pass
                else:
                    remaining_futures -= 1
            if remaining_futures == 0:
                break
        return dspikes

    def Spike(self, request, context):
        # Unpack message.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        logger.info('spike {}{}', parent_id, message_id)

        # 1. Check and build message buffer. On recursion with loops we respond
        # with a null message if the message_id has been seen already.
        self.lock.acquire()
        try:
            # Check for duplicates.
            if message_id in self.memory:
                # Return null repsonse.
                response = bittensor.proto.bittensor_pb2.SpikeResponse(
                    version=1.0,
                    source_id=source_id,
                    child_id=self.config.identity,
                    message_id=message_id,
                    payload=str.encode(''))
                return response

            # Build new message buffer.
            msg_buffer = Buffer(source_id=source_id,
                                parent_id=parent_id,
                                message_id=message_id,
                                create_time=time.time())

            self.memory[message_id] = msg_buffer

        finally:
            self.lock.release()

        # 2. Make recursive calls to downstream neighbors.
        # futures is a list of callbacks from each downstream call.
        futures = []
        for channel in self.channels:
            futures.append(self._spike_future(channel, request))

        # 3. Deserialize upstream spikes.
        uspikes = pickle.loads(request.payload)

        # 4. Fill downstream spikes.
        dspikes = [np.zeros((len(uspikes), 128)) for _ in range(self.config.k)]
        dspikes = self._fill_dspikes(dspikes, futures)

        # 5. Inference local neuron.
        lspikes = self.nucleus.spike(uspikes, dspikes)

        # 6. Sink output to memory.
        self.lock.acquire()
        try:
            self.memory[message_id].set(uspikes=uspikes,
                                        lspikes=lspikes,
                                        dspikes=dspikes)
        finally:
            self.lock.release()

        # 7. Build response.
        payload = pickle.dumps(lspikes, protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
            version=1.0,
            source_id=source_id,
            child_id=self.config.identity,
            message_id=message_id,
            payload=payload)

        # Return.
        return response

    def Grade(self, request, context):
        # Unpack request.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        ugrades = pickle.loads(request.payload)
        logger.info('grad {}{}', parent_id, message_id)

        # Check for lost or badly routed grades.
        if message_id not in self.memory:
            return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

        # Get local spikes.
        mem_buffer = self.memory[message_id]

        # Get local spikes.
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
        for channel in self.channels:
            if channel is None:
                continue
            try:
                # Build Stub and request proto.
                stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

                # Build Grade Request proto.
                request = bittensor.proto.bittensor_pb2.GradeRequest(
                    version=1.0,
                    source_id=source_id,
                    parent_id=self.config.identity,
                    message_id=message_id,
                    payload=pickle.dumps(dgrades, protocol=0))

                # Send async grade request.
                stub.Grade.future(request)

            except Exception as error:
                pass

        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

    def Learn(self):
        # Function clears the message buffer of all outdated memory objects
        # and applies gradients from memory.
        logger.info('Learn.')

        # Clean the memory.
        self.lock.acquire()
        try:
            time_now = time.time()
            to_delete = []
            for row in self.memory.values():
                if (time_now - row.create_time) > self.config.time_till_expire:
                    to_delete.append(row.message_id)

            for message_id in to_delete:
                del self.memory[message_id]

        except Exception as e:
            logger.error('Neuron failed on memory clean with Error: ' + str(e))

        finally:
            self.lock.release()

        # Apply the batch.
        logger.info('Grad queue size: {}', self.gradient_queue.qsize())
        while not self.gradient_queue.empty():
            grad = self.gradient_queue.get()
            self.nucleus.learn(grad)
