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
        return  self.identity + "-" + self.address + ":" + str(self.port)

    def __eq__(self, other):
        if not other:
            return False
        return (self.identity == other.identity)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.identity))


class Metagraph():
    def __init__(self, config):
        self.config = config
        self.cleos = Cleos(url=config.eosurl)
        self.nodes = {}
        self.pull_metagraph()
        # self.eoskey = "5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3"

    def pull_metagraph(self):
        table = self.cleos.get_table('bittensoracc', 'bittensoracc', 'peers')
        for entry in table['rows']:
            next_node = Node(entry)
            self.nodes[entry['identity']] = next_node
        logger.debug(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_rep = "\nmetagraph = {\n"
        for node in self.nodes.values():
            str_rep += ("\t" + str(node) + "\n")
        str_rep += "}."
        return  str_rep


    # def announce_self(self):
    #     logger.info('announce self')
    #     transaction = self.upsert_trx()
    #     logger.info(transaction)
    #     resp = self.cleos.push_transaction(transaction, self.eoskey, broadcast=True)
    #     logger.info(resp)
    #
    # def erase_self(self):
    #     logger.info('erase self')
    #     transaction = self.erase_trx()
    #     logger.info(transaction)
    #     resp = self.cleos.push_transaction(transaction, self.eoskey, broadcast=True)
    #     logger.info(resp)

    # def upsert_trx(self):
    #     arguments = {
    #         "user": self.config.identity,  # sender
    #         "address": self.config.address,  # receiver
    #         "port": self.config.port
    #     }
    #     payload = {
    #         "account": "bittensoracc",
    #         "name": "upsert",
    #         "authorization": [{
    #             "actor": self.config.identity,
    #             "permission": "active",
    #         }],
    #     }
    #     #Converting payload to binary
    #     data=self.cleos.abi_json_to_bin(payload['account'],payload['name'],arguments)
    #     #Inserting payload binary form as "data" field in original payload
    #     payload['data']=data['binargs']
    #     #final transaction formed
    #     trx = {"actions": [payload]}
    #     return trx
    #
    # def erase_trx(self):
    #     arguments = {
    #         "user": self.config.identity
    #     }
    #     payload = {
    #         "account": "bittensoracc",
    #         "name": "erase",
    #         "authorization": [{
    #             "actor": self.config.identity,
    #             "permission": "active",
    #         }],
    #     }
    #     #Converting payload to binary
    #     data=self.cleos.abi_json_to_bin(payload['account'],payload['name'],arguments)
    #     #Inserting payload binary form as "data" field in original payload
    #     payload['data'] = data['binargs']
    #     #final transaction formed
    #     trx = {"actions": [payload]}
    #     return trx
