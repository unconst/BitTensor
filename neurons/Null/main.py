import argparse
import bittensor
from metagraph import Metagraph
from loguru import logger
import time
import numpy
import grpc
from timeloop import Timeloop

class Neuron(bittensor.proto.bittensor_pb2_grpc.BittensorServicer):
    def __init__(self, hparams, metagraph):
        self._hparams = hparams
        self._metagraph = metagraph
        self._channels = []
        self._channel_ids = []
        self.connect()

    def connect(self):
        for node in self._metagraph.nodes.values():
            if node.identity == self._hparams.identity:
                continue
            elif node.identity not in self._channel_ids:
                address = node.address + ':' + node.port
                self._channels.append(grpc.insecure_channel(address))
                self._channel_ids.append(node.identity)

    def query(self):
        # 1. Create nounce.
        nounce = str(random.randint(0, 1000000000))

        # 2. Encode nounce and source
        source_id = self._hparams.identity
        nounce_bytes = bytes(nounce, 'utf-8')
        source_bytes = bytes(source_id, 'utf-8')
        spikes = numpy.array(['this is a test'])
        payload_bytes = pickle.dumps(spikes, protocol=0)

        # 3. Create unique message hash.
        hash = SHA256.new()
        hash.update(nounce_bytes)
        hash.update(source_bytes)
        hash.update(payload_bytes)
        message_id = hash.digest()

        # 4. Create futures.
        spike_futures = []
        for channel in self._channels:
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)
            request = bittensor.proto.bittensor_pb2.SpikeRequest(
                version=1.0,
                source_id=self._hparams.identity,
                parent_id=self._hparams.identity,
                message_id=message_id,
                payload=payload_bytes)
            spike_futures.append(stub.Spike.future(request))

        # 5. Catch future responses
        start = time.time()
        returned = [False for _ in spike_futures]
        while True:
            for i, future in enumerate(spike_futures):
                if future.done():
                    done[i] = True
            if time.time() - start < 1:
                break
            if sum(returned) == len(spike_futures):
                break
        logger.info('S: {}', returned)

        # 6. Create grad futures.
        grad_futures = []
        for channel in self._channels:
            zeros_payload = pickle.dumps(np.zeros((1, self._hparams.n_embedding)), protocol=0)
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(channel)
            request = bittensor.proto.bittensor_pb2.GradeRequest(
                version=1.0,
                source_id=self._hparams.identity,
                parent_id=self._hparams.identity,
                message_id=message_id,
                payload=zeros_payload)
            grad_futures.append(stub.Grade.future(request))

        # 7. Catch grad future responses
        start = time.time()
        returned = [False for _ in grad_futures]
        while True:
            for i, future in enumerate(grad_futures):
                if future.done():
                    done[i] = True
            if time.time() - start < 1:
                break
            if sum(returned) == len(grad_futures):
                break
        logger.info('G: {}', returned)


    def Spike(self, request, context):
        logger.info(request)
        zeros_payload = pickle.dumps(np.zeros((len(uspikes), self._hparams.n_embedding)), protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
            version=1.0,
            source_id=request.source_id,
            child_id=self._hparams.identity,
            message_id=request.message_id,
            payload=zeros_payload)

    def Grade(self, request, context):
        logger.info(request)
        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

def set_timed_loops(tl, hparams, neuron, metagraph):

    # Pull the updated graph state (Vertices, Edges, Weights)
    @tl.job(interval=timedelta(seconds=7))
    def pull_metagraph():
        metagraph.pull_metagraph()

    # Publish attributions (Edges, Weights.)
    @tl.job(interval=timedelta(seconds=3))
    def publish_attributions():
        metagraph.publish_attributions()

    # Reselect channels.
    @tl.job(interval=timedelta(seconds=10))
    def connect():
        neuron.connect()

    # Reselect channels.
    @tl.job(interval=timedelta(seconds=10))
    def connect():
        neuron.query()

def main(hparams):

    metagraph = Metagraph(hparams)
    neuron = Neuron(hparams, metagraph)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bittensor.proto.bittensor_pb2_grpc.add_BittensorServicer_to_server(neuron, server)
    server.add_insecure_port(hparams.bind_address + ":" + hparams.port)

    tl = Timeloop()
    set_timed_loops(tl, hparams, neuron, metagraph)
    tl.start(block=False)
    logger.info('Started Timers.')

    try:
        logger.info('Begin wait on main...')
        while True:
            logger.debug('heartbeat')
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        del neuron
        del metagraph

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: ' + str(e))
        del neuron
        del metagraph

if __name__ == '__main__':
    logger.debug("started neuron.")
    parser = argparse.ArgumentParser()

    # Server parameters.
    parser.add_argument(
        '--identity',
        default='abcd',
        type=str,
        help="network identity. Default identity=abcd")
    parser.add_argument(
        '--serve_address',
        default='0.0.0.0',
        type=str,
        help="Address to server neuron. Default serve_address=0.0.0.0")
    parser.add_argument(
        '--bind_address',
        default='0.0.0.0',
        type=str,
        help="Address to bind neuron. Default bind_address=0.0.0.0")
    parser.add_argument(
        '--port',
        default='9090',
        type=str,
        help="Port to serve neuron on. Default port=9090")
    parser.add_argument(
        '--eosurl',
        default='http://0.0.0.0:8888',
        type=str,
        help="Address to eos chain. Default eosurl=http://0.0.0.0:8888")
    parser.add_argument(
        '--logdir',
        default="/tmp/",
        type=str,
        help="logging output directory. Default logdir=/tmp/")
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')

    hparams = parser.parse_args()

    main(hparams)
