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

def plot():
    Gnx = nx.DiGraph()

    for r in G:
        Gnx.add_node(r)
        for e in G[r]['edges']:
            Gnx.add_edge(r, e[0], weight=int(e[1] * 100))

    pos = nx.circular_layout(G)
    nx.draw(Gnx, pos)
    labels = nx.get_edge_attributes(Gnx, 'weight')
    edge_labels=nx.draw_networkx_edge_labels(Gnx, pos, with_labels = True, edge_labels=labels, arrowsize=20, arrowstyle='->',)
    nx.draw(Gnx, pos, with_labels = True, edge_labels=edge_labels)
    plt.show()

def emit(G):

    total_stake = sum([node['stake'] for node in G.values()])

    for node_key in G.keys():
        stake_in = G[node_key]['edges'][0][1] * G[node_key]['stake']
        for other_key in G.keys():
            if other_key == node_key:
                continue
            for edge in G[other_key]['edges']:
                if edge[0] == node_key:
                    stake_in += edge[1] * G[other_key]['stake']

        # 10% stake inflation.
        emmision = (stake_in / total_stake) * 0.1

        G[node_key]['stake'] += emmision

def print_stake(G):


    print('total: ' + str(sum([node['stake'] for node in G.values()])))
    for node_key in G.keys():
        strg = str(node_key) + " " + str(G[node_key]['stake'])
        print(strg)

def print_G(G):

    for node_key in G.keys():
        print (node_key, G[node_key]['stake'], str(G[node_key]['edges']))


def main():

    for _ in range(int(sys.argv[1])):
        new_id = random_id(range(0, 100), G.keys())
        G[new_id] = {}
        G[new_id]['stake'] = float(random.randint(0, 1000))


    for id in G.keys():
        edges = []
        for _ in range(min( len(G.keys()), 3)):
            edges.append(random_id(G.keys(), edges))

        weights = [random.randint(0, 100) for _ in range(len(edges))]
        norm_weights = [w / sum(weights) for w in weights]
        attrs = list(zip(edges, norm_weights))

        G[id]['edges'] = attrs

    print_G(G)

    import time
    print_stake(G)
    for _ in range(int(sys.argv[2])):
        emit(G)
    print_stake(G)







if __name__ == "__main__":
    main()
