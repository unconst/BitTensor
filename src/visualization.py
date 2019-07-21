from __future__ import division
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy

import io
import tensorflow as tf
from PIL import Image

import os


def figure_to_buff(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  return buf

def generate_edge_weight_buffer(nodes):
    b_nodes = list(nodes.values())
    G = nx.DiGraph()

    total_stake = sum([node.stake for node in b_nodes])

    node_sizes = []
    node_labels = {}
    for node in b_nodes:
        G.add_node(node.identity)
        node_sizes.append(100 + 300*(node.stake/total_stake))
        node_labels[node.identity] = str(node.identity)

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

    for u,v,d in G.edges(data=True):
        d['weight'] = edge_colors[(u,v)]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    print (G.nodes())
    print (weights)
    print (node_sizes)

    plt.cla()
    plt.clf()
    figure = plt.figure(figsize=(20,15))
    pos = nx.layout.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color=weights, edge_cmap=plt.cm.Blues, width=5)

    edge_labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, with_labels = True, label_pos=0.7)

    for node in b_nodes:
        pos[node.identity] = pos[node.identity] + numpy.array([0, 0.1])
    labels = nx.draw_networkx_labels(G, pos, node_labels)

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    return buf
