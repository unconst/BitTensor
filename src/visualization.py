from __future__ import division
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import os
import tensorflow as tf

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

def plot_table(nodes):
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

def generate_edge_weight_buffer(nodes):
    b_nodes = list(nodes.values())
    print (b_nodes)
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
    figure = plt.figure(figsize=(20,15))

    pos = nx.layout.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color=weights, edge_cmap=plt.cm.Blues, width=5)

    edge_labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, with_labels = True, label_pos=0.3)

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
