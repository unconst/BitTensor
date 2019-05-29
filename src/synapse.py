from loguru import logger
import sys
import time
import tensorflow as tf
import proto.bolt_pb2_grpc

class BoltServicer(proto.bolt_pb2_grpc.BoltServicer):
    def __init__(self, metagraph):
        self.identity = metagraph.this_identity
        self.load_time = 20
        self.since_last_load = -1
        self.since_last_attempted_load = -1
        self.is_loading = False
        self._load_graph()

    def _load_graph(self):
        self.since_last_attempted_load = time.time()
        try:
            graph = tf.Graph()
            with graph.as_default(), tf.device('/cpu:0'):
                saver = tf.train.import_meta_graph('../checkpoints/' + self.identity + '/model.meta')
                next_session = tf.Session()
                saver.restore(next_session, tf.train.latest_checkpoint('../checkpoints/' + self.identity))
                next_session.run('init_all_tables')
                next_session.run(tf.local_variables_initializer())
                next_session.run("embedding_output:0", feed_dict={"batch_words:0": [['UNK']], 'is_training:0': False})
        except Exception as e:
            logger.error('Failed to server new graph. Exception {}', e)
            return

        logger.info('Successfully served new graph.')
        self.session = next_session
        self.since_last_load = time.time()


    def Spike(self, request, context):
        # TODO (const) The synapse should be competitively selecting which nodes
        # are allowed to query us based on the Metagraph information.
        if time.time() - self.since_last_load > self.load_time and self.is_loading == False:
            self.is_loading = True
            self._load_graph()
            self.is_loading = False

        batch_words = [[word] for word in request.string_val]
        embeddings = self.session.run("embedding_output:0", feed_dict={"batch_words:0": batch_words, 'is_training:0': False})
        embed_proto = tf.make_tensor_proto(embeddings)
        return embed_proto
