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
import tf_sentencepiece

EMBEDDING_SIZE = 128


class Neuron(bittensor.proto.bittensor_pb2_grpc.BittensorServicer):

    def __init__(self, config):
        self.config = config

        self.graph = tf.Graph()
        with self.graph.as_default():
            logger.info('Loading tensorflow hub module.')
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"
            module = hub.Module(module_url, trainable=False)

            self.text_placeholder = tf.compat.v1.placeholder(
                tf.string, shape=[None], name="text_placeholder")
            question_embeddings = module(dict(input=self.text_placeholder),
                                         signature="question_encoder",
                                         as_dict=True)

            # Projection onto EMBEDDING_SIZE
            w1 = tf.Variable(tf.random.uniform([512, EMBEDDING_SIZE], -1.0,
                                               1.0))
            b1 = tf.Variable(tf.zeros([EMBEDDING_SIZE]))
            self.output = tf.sigmoid(
                tf.matmul(question_embeddings["outputs"], w1) + b1)

            init_op = tf.group(
                [tf.global_variables_initializer(),
                 tf.tables_initializer()])
        self.graph.finalize()

        # Initialize session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(init_op)

        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor.proto.bittensor_pb2_grpc.add_BittensorServicer_to_server(
            self, self.server)
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
        inputs = pickle.loads(request.payload)
        logger.info('. {}', parent_id)

        # Inference through Google USE.
        numpy_inputs = inputs.flatten()  # [batch_size, var length]
        represenations = self.session.run(self.output,
                                          {self.text_placeholder: numpy_inputs})
        represenations = represenations.reshape(EMBEDDING_SIZE, -1)

        # Pack response.
        response_payload = pickle.dumps(represenations, protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
            child_id=self.config.identity,
            message_id=message_id,
            payload=response_payload)

        return response

    def Grade(self, request, context):
        # Pass.
        return bittensor.proto.bittensor_pb2.GradeResponse(accept=False)


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
        logger.error('Neuron stopped with interrupt on error: ' + str(e))
        tear_down(config, neuron)


if __name__ == '__main__':
    logger.debug("started neuron.")
    main()
