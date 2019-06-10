#!/usr/bin/python

import random
import sys
import networkx as nx
import matplotlib.pyplot as plt

# Map from id to edges, edges are (id, w)
G = {}

def random_id(allowed, current):
    if len(allowed) == 0:
        return

    while True:
        choice = random.choice(list(allowed))
        if choice not in current:
            return choice

def new_random(G):
    my_id = random_id(range(0, 100), G.keys())

    edges = [my_id]
    for _ in range(min( len(G.keys()), 3)):
        edges.append(random_id(G.keys(), edges))

    weights = [random.randint(0, 100) for _ in range(len(edges))]
    sum_weights = sum(weights)
    norm_weights = [w /sum_weights for w in weights]

    attrs = list(zip(edges, norm_weights))
    G[my_id] = attrs

def main():

    for _ in range(int(sys.argv[1])):
        new_random(G)

    Gnx = nx.DiGraph()

    for r in G:
        Gnx.add_node(r)
        for e in G[r]:
            Gnx.add_edge(r, e[0], weight=int(e[1] * 100))

    pos = nx.circular_layout(G)
    nx.draw(Gnx, pos)
    labels = nx.get_edge_attributes(Gnx, 'weight')
    nx.draw_networkx_edge_labels(Gnx, pos, with_labels = True, edge_labels=labels, arrowsize=20, arrowstyle='->',)
    plt.show()




if __name__ == "__main__":
    main()
