import ast
from loguru import logger
import sys

from eospy.cleos import Cleos

class Node():
    def __init__(self, entry):
        self.identity = entry['identity']
        self.address = entry['address']
        self.port = entry['port']
        self.edges = entry['edges']
        self.attributions = entry['attribution']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return  "(" + self.identity + "|" + self.address + ":" + str(self.port) + '|' + str(self.edges) + "|" + str(self.attributions) + ")"

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
        # TODO(const) this should be our own key.
        self.eoskey = "5KQwrPbwdL6PhXujxW37FSSQZ1JiwsST4cqQzDeyXtP79zkvFD3"

    def pull_metagraph(self):
        table = self.cleos.get_table('bittensoracc', 'bittensoracc', 'peers')
        for entry in table['rows']:
            next_node = Node(entry)
            self.nodes[entry['identity']] = next_node
        logger.debug(self.__str__())

    def publish_attributions(self, attributions):
        table = self.cleos.get_table('bittensoracc', 'bittensoracc', 'peers')
        for entry in table['rows']:
            next_node = Node(entry)
            self.nodes[entry['identity']] = next_node
        logger.debug(self.__str__())


    def clean_attributions(self, edge_nodes, edge_attrs):

        #logger.info(edge_nodes)
        #logger.info(edge_attrs)

        # Fill out the two lists.
        self_node = self.nodes[self.config.identity]
        edge_nodes = [self_node] + edge_nodes

        # TODO(const) sloppy.
        # edge_attrs is k + 1 vector.
        # edge_nodes is a k length vector.
        # We are going to normalize the attributions 0-1 but only over non null values.
        attr_sum = 0
        for i in range(self.config.k + 1):
            if edge_nodes[i] is not None:
                attr_sum += edge_attrs[i]

        result_ids = []
        result_attrs = []
        for i in range(self.config.k + 1):
            if edge_nodes[i] is not None:
                result_ids.append(edge_nodes[i].identity)
                result_attrs.append(edge_attrs[i]/attr_sum)

        return result_ids, result_attrs

    def set_attributions(self, edge_nodes, attributions):
        self.edge_ids, self.edge_attrs = self.clean_attributions(edge_nodes, attributions)

    def publish_attributions(self):

        #logger.info('publish attribuions. {} {}', edge_ids, edge_attrs )
        transaction = self.publish_attributions_trx(self.edge_ids, self.edge_attrs)

        #logger.info(transaction)
        resp = self.cleos.push_transaction(transaction, self.eoskey, broadcast=True)
        #logger.info(resp)

    def publish_attributions_trx(self, edge_ids, edge_attrs):

        arguments = {
            "user": self.config.identity,
            "edges": edge_ids,
            "attribution": edge_attrs
        }
        payload = {
            "account": "bittensoracc",
            "name": "grade",
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
