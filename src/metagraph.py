from loguru import logger
import sys

class Metagraph():
    def __init__(self, argv):
        # TODO(const) This class implements the connection to our blockchain.
        # Rather than take state from command line args, it should read a
        # smart contract.
        self.this_address = str(argv[1])
        self.this_identity = self.this_address.split(']:')[1]
        self.remote_neurons = [addr for addr in argv[2:]]
