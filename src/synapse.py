from loguru import logger
import sys
import time
import tensorflow as tf
import proto.bolt_pb2_grpc

# TODO (const): Rate limit and block ip.

class BoltServicer(proto.bolt_pb2_grpc.BoltServicer):
    def __init__(self, config, metagraph):
        self.identity = config.identity
        self.config = config
        self.metegraph = metagraph
        self.load_graph()

    def load_graph(self):
        try:
            graph = tf.Graph()
            with graph.as_default(), tf.device('/cpu:0'):
                saver = tf.train.import_meta_graph('data/' + self.identity + '/model.meta')
                next_session = tf.Session()
                saver.restore(next_session, tf.train.latest_checkpoint('data/' + self.identity))
                next_session.run('init_all_tables')
                next_session.run(tf.local_variables_initializer())
                next_session.run("embedding_output:0", feed_dict={"batch_words:0": [['UNK']], 'is_training:0': False})
        except Exception as e:
            logger.error('failed to server new graph. Exception {}', e)
            return

        logger.debug('served new graph.')
        self.session = next_session

    def Spike(self, request, context):
        # TODO (const) The synapse should be competitively selecting which nodes
        # are allowed to query us based on the Metagraph information.
        batch_words = [[word] for word in request.string_val]
        embeddings = self.session.run("embedding_output:0", feed_dict={"batch_words:0": batch_words, 'is_training:0': False})
        embed_proto = tf.make_tensor_proto(embeddings)
        return embed_proto
