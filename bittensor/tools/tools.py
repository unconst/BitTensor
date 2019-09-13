import argparse
from loguru import logger
from eospy.cleos import Cleos
from metagraph import Node
import visualization

parser = argparse.ArgumentParser(description='TF graph client args.')
parser.add_argument('--command', default="info")
parser.add_argument('--eosurl', default="http://host.docker.internal:8888")

def _make_plot_table(nodes):
    b_nodes = list(nodes.values())
    G = nx.DiGraph()

    total_stake = sum([node.stake for node in b_nodes])

    # Build node sizes in proportion to stake held within the graph.
    node_sizes = []
    node_labels = {}
    for node in b_nodes:
        G.add_node(node.identity)
        node_sizes.append(25 + 500*(node.stake/total_stake))
        node_labels[node.identity] = str(node.identity)

    # Edge colors (alphas and weight) reflect attribution wieghts of each
    # connection.
    edge_colors = {}
    edge_labels = {}
    for node in b_nodes:
        for edge in node.edges:
            if (node.identity, edge['first']) not in edge_labels:
                G.add_edge(node.identity, edge['first'])
                edge_colors[(node.identity, edge['first'])] = float(edge['second'])
                if node.identity != edge['first']:
                    edge_labels[(node.identity, edge['first'])] = "%.3f" % float(edge['second'])
                else:
                    edge_labels[(node.identity, edge['first'])] = ""

    # Set edge weights.
    for u,v,d in G.edges(data=True):
        d['weight'] = edge_colors[(u,v)]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    # Clear Matplot lib buffer and create new figure.
    plt.cla()
    plt.clf()
    figure = plt.figure(figsize=(15,8))

    pos = nx.layout.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color=weights, edge_cmap=plt.cm.Blues, width=5)

    edge_labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, with_labels = True, label_pos=0.3)

    for node in b_nodes:
        pos[node.identity] = pos[node.identity] + numpy.array([0, 0.1])
    labels = nx.draw_networkx_labels(G, pos, node_labels)

    plt.show(figure)

def plot_table(table):
    nodes = {}
    for entry in table['rows']:
        next_node = Node(entry)
        nodes[entry['identity']] = next_node
    _make_plot_table(nodes)

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
