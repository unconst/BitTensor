import bittensor

from config import Config

from concurrent import futures
import grpc
from loguru import logger
import numpy
import pickle

import time
import tensorflow as tf
import tensorflow_hub as hub


EMBEDDING_SIZE=128

class Neuron(bittensor.proto.bolt_pb2_grpc.BoltServicer):

    def __init__(self, config):
        self.config = config

        # Load ELMO model.
        logger.info('Loading tensorflow hub module.')
        logger.info('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')
        tf.compat.v1.enable_eager_execution()
        module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
        self.embed = hub.KerasLayer(module_url)
        logger.info('done.')

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
        inputs = pickle.loads(request.payload)

        # Inference through EMLO.
        embeddings = numpy.array(self.embed(inputs.flatten())).reshape(EMBEDDING_SIZE, -1)

        # Pack response.
        response_payload = pickle.dumps(embeddings, protocol=0)
        response = bittensor.proto.bolt_pb2.SpikeResponse(
                        responder_identity = self.config.identity,
                        message_identity = message_id,
                        payload = response_payload)

        return response


    def Grade(self, request, context):
        # Pass.
        return bittensor.proto.bolt_pb2.GradeResponse(accept=True)

def main():
    config = Config()

    neuron = Neuron(config)

    neuron.serve()

    def tear_down(_config, _neuron):
        logger.debug('tear down.')
        del _neuron
        del _config

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(config, neuron)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: '+ str(e))
        tear_down(config, neuron)

if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
