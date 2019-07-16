import ast
from loguru import logger
import sys

import eospy.keys
from eospy.cleos import Cleos

class Node():
    def __init__(self, entry):
        # EOS account name.
        self.identity = entry['identity']
        # Network Stake.
        self.stake = entry['stake']
        # Last emit.
        self.last_emit = entry['last_emit']
        # IP address.
        self.address = entry['address']
        # Port number.
        self.port = entry['port']
        # List of tuples (edge name, edge weight)
        self.edges = entry['edges']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        edge_str = []
        for el in self.edges :
            edge_str.append((el['first'], el['second']))
        edge_str = str(edge_str)
        return  "( " + self.identity + " | " + str(self.stake) + " | " + str(self.last_emit) + " | " + self.address + ":" + str(self.port) + ' | ' + edge_str + " )"

    def __eq__(self, other):
        if not other:
            return False
        return (self.identity == other.identity)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.identity))


# TODO (const): Pull Token supply.
# TODO (const): Call Token Emit.
# TODO (const): Calculate attribution derivatives for synapse prunning.

class Metagraph():
    def __init__(self, config):
        self.config = config
        self.cleos = Cleos(url=config.eosurl)
        self.nodes = {}
        self.pull_metagraph()
        self.attributions = [(config.identity, 1.0)]
        # TODO(const) this should be our own key. NOT EOSMAIN.
        self.eoskey = "5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3"

    def pull_metagraph(self):
        table = self.cleos.get_table('bittensoracc', 'bittensoracc', 'metagraph')
        for entry in table['rows']:
            next_node = Node(entry)
            self.nodes[entry['identity']] = next_node
        logger.debug(self.__str__())


    # # TODO(const): set attributions.
    # def set_attributions():
    #     pass
    #
    # # TODO(const): refresh graph.
    # # Read the EOS blockchain for an updated metagraph state.
    # def refresh():
    #     pass
    #
    # # TODO(const): emit token.
    # # Call EOS emit to mint new tokens and post new attribution values.
    # def emit():
    #     pass
    #
    # # TODO(const): subscribe neuron.
    # # Move this neuron onto the EOS metagraph.
    # def subscribe():
    #     pass
    #
    # # TODO(const): unsubscribe neuron.
    # # Remove this neuron from the EOS metagraph.
    # def unsubscribe():
    #     pass

    # Push attribution scores.
    def publish_attributions(self):
        logger.debug('Publish attributions: ' + str(self.attributions))
        transaction = self.publish_attributions_trx()
        resp = self.cleos.push_transaction(transaction, self.eoskey, broadcast=True)

    def publish_attributions_trx(self):
        arguments = {
            "this_user": self.config.identity,
            "this_edges": [(attr[0], float(attr[1])) for attr in self.attributions],
        }
        payload = {
            "account": "bittensoracc",
            "name": "emit",
            "authorization": [{
                "actor": self.config.identity,
                "permission": "active",
            }],
        }
        #Converting payload to binary
        data=self.cleos.abi_json_to_bin(payload['account'],payload['name'],arguments)
        #Inserting payload binary form as "data" field in original payload
        payload['data']=data['binargs']
        #final transaction formed
        trx = {"actions": [payload]}
        return trx

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_rep = "\nmetagraph = {\n"
        for node in self.nodes.values():
            str_rep += ("\t" + str(node) + "\n")
        str_rep += "}."
        return  str_rep

#
# Error: {'code': 500,
#         'message': 'Internal Service Error',
#         'error': {
#             'code': 3050002,
#             'name': 'invalid_action_args_exception',
#             'what': 'Invalid Action Arguments',
#             'details': [
#                 {
#                     'message':
#                         '\'{
#                             "user":"zragtdp",
#                             "edges":"[\'zragtdp\', \'test\']",
#                             "attributions":"[0.3053072399080243, 2.7059856455960176]"}\
#                             ' is invalid args for action \'grade\' code \'bittensoracc\'.
#                             'expected \'
#                             '[{"name":"user","type":"name"},{"name":"edges","type":"name[]"},
#                             '{"name":"attribution","type":"float32[]"}]'
#
#                             '\'', 'file': 'chain_plugin.cpp', 'line_number': 1836, 'method': 'abi_json_to_bin'},
#                             {'message': 'Invalid cast from string_type to Array', 'file': 'variant.cpp', 'line_number': 545, 'method': 'get_array'},
#                             {'message': '', 'file': 'abi_serializer.cpp', 'line_number': 496, 'method': '_variant_to_binary'},
#                             {'message': '', 'file': 'abi_serializer.cpp', 'line_number': 496, 'method': '_variant_to_binary'},
#                             {'message': '', 'file': 'abi_serializer.cpp', 'line_number': 510, 'method': '_variant_to_binary'}, {
#                             'message': 'code: bittensoracc, action: grade,
#                             'args: {"user":"zragtdp","edges":"[\'zragtdp\', \'test\']","attributions":"[0.3053072399080243, 2.7059856455960176]"}', 'file': 'chain_plugin.cpp', 'line_number': 1842, 'method': 'abi_json_to_bin'}]}}


    #
    # def erase_self(self):
    #     logger.info('erase self')
    #     transaction = self.erase_trx()
    #     logger.info(transaction)
    #     resp = self.cleos.push_transaction(transaction, self.eoskey, broadcast=True)
    #     logger.info(resp)

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
