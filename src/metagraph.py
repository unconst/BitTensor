from loguru import logger
import sys

from eospy.cleos import Cleos


class Node():
    def __init__(self, entry):
        self.identity = entry['identity']
        self.address = entry['address']
        self.port = entry['port']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return  "{identity: " + self.identity + " " +\
                "address: " + self.address + " " +\
                "port: " + self.port + "}"

class Metagraph():
    def __init__(self, config):
        self.config = config
        self.cleos = Cleos(url=config.eosurl)
        self.nodes = {}
        self.pull_metagraph()

    def pull_metagraph(self):
        logger.info('pull_metagraph')
        table = self.cleos.get_table('bittensoracc', 'bittensoracc', 'peers')
        for entry in table['rows']:
            self.nodes[entry['identity']] = Node(entry)
        logger.info(self.nodes)
