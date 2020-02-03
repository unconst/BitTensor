import bittensor
import visualizer

from Crypto.Hash import SHA256
from concurrent import futures
import grpc
from loguru import logger
import numpy as np
import math
import pickle
import random
import time
from threading import Lock
import threading
import queue

import tensorflow as tf
from io import StringIO
import matplotlib.pyplot as plt

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

class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)


    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class Neuron(bittensor.proto.bittensor_pb2_grpc.BittensorServicer):

    def __init__(self, config, nucleus, metagraph):
        self.config = config
        self.nucleus = nucleus
        self.metagraph = metagraph
        self._is_training = True
        self.current_stats = {
            'gs': None, 
            'step': None, 
            'mem': None,
            'loss': None,
            'metrics': None,
            'scores': None
        }

        self.lock = Lock()
        self.memory = {}

        # Tensorboard logger
        self._tblogger = TBLogger(self.config.logdir)

        # Metrics
        self._metrics = {}

        # child scores.
        self._scores = [0 for _ in range(self.config.n_children + 1)]

        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor.proto.bittensor_pb2_grpc.add_BittensorServicer_to_server(
            self, self.server)
        visualizer.proto.visualizer_pb2_grpc.add_VisualizerServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.server_address)

        self.channels = [None for _ in range(self.config.n_children)]
        self.channel_ids = [None for _ in range(self.config.n_children)]
        self.connect()

    def connect(self):
        for i in range(self.config.n_children):
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
        self._stop_training()
        logger.debug('Stopped Serving Neuron at: {}.', self.server_address)

    def serve(self):
        self.server.start()
        self._start_training()
        logger.debug('Started Serving Neuron at: {}.', self.server_address)

    def _spike_future(self, channel, source_id, message_id, payload):
        if channel == None:
            return None
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

            # Create spike request proto.
            request = bittensor.proto.bittensor_pb2.SpikeRequest(
                version=1.0,
                source_id=source_id,
                parent_id=self.config.identity,
                message_id=message_id,
                payload=payload)

            # Send TCP spike request with futures callback.
            return stub.Spike.future(request)
        except:
            return None

    def _fill_dspikes(self, dspikes, futures):
        while True:
            remaining_futures = sum(1 for _ in filter(None.__ne__, self.channel_ids))
            for i in range(self.config.n_children):
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
            if remaining_futures == 0:
                break
        return dspikes


    def Spike(self, request, context):
        # Unpack message.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        uspikes = pickle.loads(request.payload)
        if parent_id not in self._metrics:
            self._metrics[parent_id] = 0
        self._metrics[parent_id] += 1

        # 1. Check and build message buffer. On recursion with loops we respond
        # with a null message if the message_id has been seen already.
        self.lock.acquire()
        try:
            # Check for duplicates.
            if message_id in self.memory:
                # Return null repsonse.
                zeros_payload = pickle.dumps(np.zeros((len(uspikes), self.config.n_embedding)), protocol=0)
                response = bittensor.proto.bittensor_pb2.SpikeResponse(
                    version=1.0,
                    source_id=source_id,
                    child_id=self.config.identity,
                    message_id=message_id,
                    payload=zeros_payload)
                return response

            # Build new message buffer.
            msg_buffer = Buffer(source_id=source_id,
                                parent_id=parent_id,
                                message_id=message_id,
                                create_time=time.time())

            self.memory[message_id] = msg_buffer

        finally:
            self.lock.release()

        # 4. Fill downstream spikes. (all zeros)
        dspikes = [np.zeros((len(uspikes), 128)) for _ in range(self.config.n_children)]

        # 5. Inference local neuron.
        lspikes = self.nucleus.spike(uspikes, dspikes, use_synthetic=True)

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
        dgrades = self.nucleus.grade(ugrades, uspikes, dspikes)

        # delete memory:
        del self.memory[message_id]

        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

    def _start_training(self):
        self._is_training = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _stop_training(self):
        self._is_training = False
        self._thread.join()

    def _run(self):

        step = 0
        last_log_step = 0
        last_log_time = time.time()
        while self._is_training:
            step+=1

            # 1. Next training batch.
            nounce = str(random.randint(0, 1000000000))
            spikes, targets = self.nucleus.next_batch(self.config.batch_size)

            # 2. Encode nounce and source
            source_id = self.config.identity
            nounce_bytes = bytes(nounce, 'utf-8')
            source_bytes = bytes(source_id, 'utf-8')
            payload_bytes = pickle.dumps(spikes, protocol=0)

            # 3. Create unique message hash.
            hash = SHA256.new()
            hash.update(nounce_bytes)
            hash.update(source_bytes)
            hash.update(payload_bytes)
            message_id = hash.digest()

            # 3. Make recursive calls to downstream neighbors.
            # futures is a list of callbacks from each downstream call.
            futures = []
            for channel in self.channels:
                futures.append(self._spike_future(channel, source_id, message_id, payload_bytes))

            # 4. Fill responses.
            dspikes = [np.zeros((self.config.batch_size, self.config.n_embedding)) for _ in range(self.config.n_children)]
            dspikes = self._fill_dspikes(dspikes, futures)

            # 5. Train local model.
            dgrads, loss, scores = self.nucleus.train(spikes, dspikes, targets)

            # 6. Send downstream grads.
            for i, channel in enumerate(self.channels):
                if channel is None:
                    continue

                # Build Stub and request proto.
                stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)

                # Build Grade Request proto.
                request = bittensor.proto.bittensor_pb2.GradeRequest(
                    version=1.0,
                    source_id=source_id,
                    parent_id=self.config.identity,
                    message_id=message_id,
                    payload=pickle.dumps(dgrads[i][0], protocol=0))

                # Send async grade request.
                stub.Grade.future(request)

            # 7. Average score values.
            for i, score in enumerate(scores):
                prev_score = (self._scores[i] * (1 - self.config.score_ema))
                next_score = score * (self.config.score_ema)
                self._scores[i] = prev_score + next_score

            # 8. Logs
            if step % 50 == 49:
                time_now = time.time()

                # Clean mem.
                for key in self.memory:
                    val = self.memory[key]
                    if val.create_time - time_now > 5:
                        del self.memory[key]

                # global step calulcation and log.
                steps_since_last_log = last_log_step - step
                secs_since_last_log = last_log_time - time_now
                last_log_step = step
                last_log_time = time_now
                gs = steps_since_last_log / secs_since_last_log
                self._tblogger.log_scalar("gs", gs, step)

                # Log memory size.
                self._tblogger.log_scalar("mem", len(self.memory), step)

                # Log loss
                self._tblogger.log_scalar("loss", loss, step)

                # Metrics
                for idn in self._metrics.keys():
                    self._tblogger.log_scalar(idn, self._metrics[idn], step)

                # Clean and average the scores.
                clean_scores = self._clean_scores(self.config.identity, 0.5, self.channel_ids, self._scores)

                # Post scores to tb.
                for idn, score in clean_scores:
                    self._tblogger.log_scalar(idn, score, step)

                # Record stats for later consumption by visualizer
                self.current_stats['gs'] = gs
                self.current_stats['step'] = step
                self.current_stats['mem'] = len(self.memory)
                self.current_stats['loss'] = loss
                self.current_stats['metrics'] = self._metrics
                self.current_stats['score'] = clean_scores

                self.metagraph.attributions = clean_scores
                logger.info('gs {} mem {} loss {} scores {}', gs, len(self.memory), loss, clean_scores)

    def _clean_scores(self, in_id, in_loop, channel_ids, scores):

        # Get non-null scores.
        non_null_scores = []
        non_null_ids = []
        for i, idn in enumerate(channel_ids):
            if idn != None:
                non_null_scores.append(scores[i])
                non_null_ids.append(idn)

        # Null children return self loop = 1.
        if len(non_null_ids) == 0:
            return [(in_id, 1.0)]

        # Up shift.
        min_non_null = abs(min(non_null_scores))
        non_null_scores = [score + min_non_null*2 for score in non_null_scores]

        # Normalize child scores.
        norm_child_scores = []
        if sum(non_null_scores) != 0:
            norm_child_scores = [ score * in_loop / sum(non_null_scores) for score in non_null_scores]
        else:
            norm_child_scores = [ in_loop * (1 / len(non_null_scores)) for _ in non_null_scores]

        # Zip it. Zip it good.
        return_val = [(in_id, in_loop)]
        return_val += list(zip(non_null_ids, norm_child_scores))
        return return_val
    
    def Report(self, request, context):
        logger.info("Reporting!")
        source_id = request.source_id
        payload_bytes = pickle.dumps(self.current_stats)
        
        return visualizer.visualizer_pb2.ReportResponse(
            version=1.0, 
            source_id="2", 
            payload=payload_bytes
        )
