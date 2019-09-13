import bittensor

from loguru import logger
import numpy as np
import pickle
import sys
import time
import tensorflow as tf

# TODO (const): Rate limit and block ip.
class BoltServicer(bittensor.proto.bolt_pb2_grpc.BoltServicer):
    def __init__(self, config, metagraph):
        """ Serves the inference graph for use by the network.
        Graphs being produced in trainging are served by the Synapse object.
        The Synapse is responsible for upstream connections, for rate limiting,
        and through this, negotiating for higher attribution within the Metagraph.

        Since the Synapse object is merely serving the inference graph,
        it is detached from the Nucleus and Dendrite during training,
        only communicating with these objects by pulling the latest and
        best inference graph from the storage directory.

        identity:
            This node's identity within the network (tied to an EOS account)
        config:
            A Bittensor config object.
        metagraph:
            A Metagraph object which maintains state about the bittensor network.
        """
        logger.debug('Init Synapse.')
        self.identity = config.identity
        self.config = config
        self.metegraph = metagraph
        self.load_graph()

    def load_graph(self):
        """ Loads the latest checkpoint from the neuron root dir.
        Args:
        Returns:
        """
        logger.debug('Trying to serve graph on Synapse ...')
        try:
            graph = tf.Graph()
            with graph.as_default(), tf.device('/cpu:0'):
                saver = tf.compat.v1.train.import_meta_graph('data/' + self.identity + '/model.meta')
                next_session = tf.compat.v1.Session()
                saver.restore(next_session, tf.compat.v1.train.latest_checkpoint('data/' + self.identity))
                next_session.run('init_all_tables')
                next_session.run(tf.compat.v1.local_variables_initializer())
                next_session.run("embedding_output:0",
                        feed_dict={
                                "inference_batch_words:0": [['UNK']], # Inference.
                                'is_training:0': False
                                })
        except Exception as e:
            logger.error('Failed to server new graph. Exception {}', e)
            raise Exception(e)

        logger.debug('Served graph on Synapse.')
        self.session = next_session

    def Spike(self, request, context):
        """ GRPC request handler for message Spike; Runs tensor request through the graph.
        Args:
            request: A bolttensorflow.TensorProto proto as defined in src/proto/bolt.proto
                containing the incoming words. The proto should be a unidimensional array of strings.
                theses strings are words to be embedded.
            context: A GPRC message context.
        Returns:
            response: A SpikeResponse proto containing identity,
                message identifier, and the embedded words (payload)
                as outputed by running the session graph.
        """
        # TODO (const) The synapse should be competitively selecting which nodes
        # are allowed to query us based on the Metagraph information.
        batch_words = pickle.loads(request.payload)
        embeddings = self.session.run("embedding_output:0",
                            feed_dict={
                                    "inference_batch_words:0": batch_words.tolist(), # Inference.
                                    'is_training:0': False
                                })
        payload = pickle.dumps(embeddings, protocol=0)
        response = bittensor.proto.bolt_pb2.SpikeResponse(
                        responder_identity = self.config.identity,
                        message_identity = request.message_identity,
                        payload = payload)
        return response


    def Grade(self, request, context):
        """ GRPC request handler for message Grade; Accepts a gradient message.
        Args:
            request: A grade message proto as defined in src/proto/bolt.proto
                containing the request identity, message identifier, and payload.
                The payload should be interpreted as a gradients w.r.t the input payload
                theses strings are words to be embedded.
            context: A GPRC message context.
        Returns:
            response: A GradeResponse proto containing an accepted message.
        """
        # TODO(const) this should append gradient messages to a training queue.
        return bittensor.proto.bolt_pb2.GradeResponse(accept=True)
        # pass
        # # TODO (const) The synapse should be competitively selecting which nodes
        # # are allowed to query us based on the Metagraph information.
        # batch_words = [[word] for word in request.string_val]
        # embeddings = self.session.run("embedding_output:0",
        #                     feed_dict={
        #                             "inference_batch_words:0": batch_words, # Inference.
        #                             'is_training:0': False
        #                         })
        # embed_proto = tf.compat.v1.make_tensor_proto(embeddings)
        # return proto.bolt_pb2.GradeResponse(accept=True)
