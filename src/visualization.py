from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
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
  # # Convert PNG buffer to TF image
  # image = tf.image.decode_png(buf.getvalue(), channels=4)
  # # Add the batch dimension
  # image = tf.expand_dims(image, 0)
  # return image

# def load_image_to_tensor(config):
#     # This portion is part of my test code
#     print (os.getcwd())
#
#     try:
#         byteImg = Image.open(str(os.getcwd()) + "/" + config.logdir + str("/metagraph_state.png")).tobytes()
#     except:
#         return None
#
#     # Non test code
#     buf = io.BytesIO(byteImg)
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#
#     # Add the batch dimension
#     image = tf.expand_dims(image, 0)
#     return image


def generate_edge_weight_plot(nodes):
    print (nodes)
    b_nodes = list(nodes.values())
    G = nx.DiGraph()

    node_sizes = []
    node_labels = {}
    for node in b_nodes:
        G.add_node(node.identity)
        node_sizes.append(node.stake)
        node_labels[node.identity] = str(node.identity)

    edge_colors = []
    edge_alphas = []
    edge_labels = {}
    for node in b_nodes:
        for edge in node.edges:
            G.add_edge(node.identity, edge['first'])
            edge_colors.append(float(edge['second']))
            edge_alphas.append(float(edge['second']))
            edge_labels[(node.identity, edge['first'])] = "%.3f" % float(edge['second'])

    print (edge_labels)
    print (G.edges())

    figure = plt.figure(figsize=(20,15))

    pos = nx.layout.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2)

    edge_labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, with_labels = True)

    print (pos)

    for node in b_nodes:
        pos[node.identity] = pos[node.identity] + numpy.array([0, 0.1])
    labels = nx.draw_networkx_labels(G, pos, node_labels)


    # set alpha value for each edge
    for i in range(len(b_nodes)):
        edges[i].set_alpha(edge_alphas[i])


    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()

    return figure
