import bittensor

from config import Config
from metagraph import Metagraph
from dendrite import Dendrite
from nucleus import Nucleus
from neuron import Neuron


from Crypto.Hash import SHA256
from datetime import timedelta
import grpc
from loguru import logger
import pickle
import numpy as np
import random
import time
from timeloop import Timeloop

def set_timed_loops(tl, config, neuron):

    #Test self.
    # @tl.job(interval=timedelta(seconds=1))
    # def test():
    #     channel = grpc.insecure_channel(config.serve_address + ":" + config.port)
    #
    #     # Inc message id.
    #     message_id = random.randint(0, 1000000)
    #
    #     # Make request.
    #     spikes = np.array([['apples']])
    #     stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)
    #
    #     # Build hash.
    #     hash = SHA256.new()
    #     hash.update(config.identity.encode())
    #     hash.update(spikes.tobytes())
    #     message_hash = hash.digest()
    #
    #     # Build request.
    #     request =  bittensor.proto.bolt_pb2.SpikeRequest()
    #     request.sender_identity = config.identity
    #     request.message_identity = message_hash
    #     request.payload = pickle.dumps(spikes,  protocol=0)
    #
    #     # Send Spike.
    #     try:
    #         response = stub.Spike(request)
    #         response = pickle.loads(response.payload).reshape(1, 128)
    #
    #     except Exception as e:
    #         logger.error(str(e))
    #
    #     # Make grad request.
    #     grad = np.zeros((1, 128))
    #     stub = bittensor.proto.bolt_pb2_grpc.BoltStub(channel)
    #
    #     # Build hash.
    #     hash = SHA256.new()
    #     hash.update(config.identity.encode())
    #     hash.update(spikes.tobytes())
    #     message_hash = hash.digest()
    #
    #     request = bittensor.proto.bolt_pb2.GradeRequest()
    #     request.sender_identity = config.identity
    #     request.message_identity = message_hash
    #     request.payload = pickle.dumps(grad,  protocol=0)
    #
    #     # Send grade request.
    #     try:
    #         stub.Grade(request)
    #     except Exception as e:
    #         logger.error(str(e))


    # Apply a gradient step.
    @tl.job(interval=timedelta(seconds=3))
    def learn():
        neuron.Learn()

def main():

    config = Config()

    metagraph = Metagraph(config)

    dendrite = Dendrite(config)

    nucleus = Nucleus(config)

    neuron = Neuron(config, dendrite, nucleus, metagraph)

    neuron.serve()

    # Start timed calls.
    tl = Timeloop()
    set_timed_loops(tl, config, neuron)
    tl.start(block=False)
    logger.info('Started Timers.')

    def tear_down(_config, _neuron, _dendrite, _nucleus, _metagraph):
        logger.debug('tear down.')
        del _neuron
        del _dendrite
        del _nucleus
        del _metagraph
        del _config

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(config, neuron, dendrite, nucleus, metagraph)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: '+ str(e))
        tear_down(config, neuron, dendrite, nucleus, metagraph)

if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
