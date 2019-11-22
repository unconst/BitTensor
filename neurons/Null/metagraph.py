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
        for el in self.edges:
            edge_str.append((el['first'], "%.4f" % float(el['second'])))
        edge_str = str(edge_str)
        return "( " + self.identity + " | " + str(self.stake) + " | " + str(
            self.last_emit) + " | " + self.address + ":" + str(
                self.port) + ' | ' + edge_str + " )"

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

    def get_my_stake(self):
        return int(self.nodes[self.config.identity].stake)

    # TODO(const): pull this from the eos chain under the var 'total stake'
    # instead of doing a sum.
    def get_total_stake(self):
        return int(sum([node.stake for node in self.nodes.values()]))

    def pull_metagraph(self):
        table = self.cleos.get_table('bittensoracc', 'bittensoracc',
                                     'metagraph')
        for entry in table['rows']:
            next_node = Node(entry)
            self.nodes[entry['identity']] = next_node
        logger.debug(self.__str__())

    # Push attribution scores.
    def publish_attributions(self):
        transaction = self.publish_attributions_trx()
        try:
            # TODO (const) Rewrite the cleos library for our selves.
            resp = self.cleos.push_transaction(transaction,
                                               self.eoskey,
                                               broadcast=True)
        except:
            try:
                eoskey = eospy.keys.EOSKey(self.eoskey)
                resp = self.cleos.push_transaction(transaction,
                                                   eoskey,
                                                   broadcast=True)
            except Exception as e:
                logger.error('Failed to publish transaction', e)

    def publish_attributions_trx(self):
        arguments = {
            "this_user":
                self.config.identity,
            "this_edges": [
                (attr[0], float(attr[1])) for attr in self.attributions
            ],
        }
        payload = {
            "account":
                "bittensoracc",
            "name":
                "emit",
            "authorization": [{
                "actor": self.config.identity,
                "permission": "active",
            }],
        }
        #Converting payload to binary
        data = self.cleos.abi_json_to_bin(payload['account'], payload['name'],
                                          arguments)
        #Inserting payload binary form as "data" field in original payload
        payload['data'] = data['binargs']
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
        return str_rep
