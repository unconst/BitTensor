import argparse
from loguru import logger
from eospy.cleos import Cleos
from metagraph import Node
import visualization

parser = argparse.ArgumentParser(description='TF graph client args.')
parser.add_argument('--command', default="info")
parser.add_argument('--eosurl', default="http://host.docker.internal:8888")

def plot_table(table):
    nodes = {}
    for entry in table['rows']:
        next_node = Node(entry)
        nodes[entry['identity']] = next_node
    visualization.plot_table(nodes)

if __name__ == "__main__":
    args = parser.parse_args()
    cleos = Cleos(url=args.eosurl)
    if args.command == "info":
        logger.info(cleos.get_info())
    elif args.command == "table":
        cleos.get_info()
        table = cleos.get_table('bittensoracc', 'bittensoracc', 'metagraph')
        plot_table(table)
    else:
        logger.info('Command not found.')
