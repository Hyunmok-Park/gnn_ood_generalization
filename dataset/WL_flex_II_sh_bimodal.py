import argparse
import collections
from itertools import product
from networkx.utils import py_random_state
import networkx as nx
from tqdm import tqdm
import numpy as np
import pickle
import os
import random
import math

def compute_count(channel, group):
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out


@py_random_state(3)
def ws_graph(n, mean, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """

    # Generate G1
    k1 = n - 1
    n1 = n
    edge_num = int(round(k1 * n1 / 2))
    count = compute_count(edge_num, n1)
    # print(count)
    G1 = nx.Graph()
    for i in tqdm(range(n1), leave=False):
        source = [i] * count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node % n1 for node in target]
        # print(source, target)
        G1.add_edges_from(zip(source, target))

    # Generate G2
    k2 = mean
    n2 = 100 - n
    edge_num = int(round(k2 * n2 / 2))
    count = compute_count(edge_num, n2)
    # print(count)
    G2 = nx.Graph()
    for i in tqdm(range(n2), leave=False):
        source = [i] * count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node % n2 for node in target]
        # print(source, target)
        G2.add_edges_from(zip(source, target))

    # connect G1, G2
    mapping = dict(zip(G2, [i + len(G1.nodes()) for i in range(len(G2.nodes()))]))
    G2 = nx.relabel_nodes(G2, mapping)
    G = nx.compose(G1, G2)

    nodes_to_connect = random.sample(G2.nodes(), len(G1.nodes()))
    for i, j in zip(G1.nodes(), random.sample(nodes_to_connect, len(nodes_to_connect))):
        G.add_edge(i, j)

    # rewire edges from each node
    nodes = list(G.nodes())
    n=100
    edge_list = np.array(G.edges())
    for i in tqdm(range(n), leave=False):
        u = i
        target_index_list_candi = list(np.where(edge_list[:, 0] == i)[0])
        number_to_sample = math.ceil(0.5 * len(target_index_list_candi))
        target_index = random.sample(target_index_list_candi, number_to_sample)
        target = edge_list[target_index][:, 1]
        edge_list = np.delete(edge_list, target_index, axis=0)

        for v in target:
            if seed.random() < p:
                w = seed.choice(nodes)
#                 print(w)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    # print(w)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)

    for i in edge_list:
        u = i[1]
        v = i[0]
        if seed.random() < p:
            w = seed.choice(nodes)
            #                 print(w)
            # Enforce no self-loops or multiple edges
            while w == u or G.has_edge(u, w):
                w = seed.choice(nodes)
                # print(w)
                if G.degree(u) >= n - 1:
                    break  # skip this rewiring
            else:
                G.remove_edge(u, v)
                G.add_edge(u, w)

    return G

@py_random_state(4)
def connected_ws_graph(n, mean, p, tries=1000, seed=1):
    """Returns a connected ws-flex graph.
    """
    for i in tqdm(range(tries), desc='try', leave=False):
        # print("try", i)
        # seed is an RNG so should change sequence each call
        G = ws_graph(n, mean, p, seed)
        if nx.is_connected(G):
            return G
    print("max try: p{}_seed{}".format(p,seed))

def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # parser.add_argument('--node', type=int, default=1, help='node')
    # # parser.add_argument('--degree', type=int, default=1, help='degree')
    # # parser.add_argument('--prob', type=int, default=1, help='prob')
    # # parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--index', type=int, default=1, help='index')
    args = parser.parse_args()

    for node in range(3,50):
        prob_space = np.linspace(0, 1, 100)**2
        seed_space = [i for i in range(30)]

        space = list(product(prob_space, seed_space))
        ii = args.index
        prop_list = []
        name_list = []
        dic = {}

        for i in space[750 * ii: 750 * (ii + 1)]:
            for mean in range(node, 100-node):
                graph = {}
                G = connected_ws_graph(node, mean, i[0], 500, i[1])
                assert(len(nx.nodes(G)) == 100)
                if G is not None:
                    J = nx.adjacency_matrix(G)

                    _, deg, _ = degree(G)
                    graph['mean'] = np.mean(deg)
                    graph['std'] = np.std(deg)
                    graph['max'] = np.max(deg)
                    graph['min'] = np.min(deg)
                    graph['cc'] = nx.average_clustering(G)
                    graph['path'] = nx.average_shortest_path_length(G)
                    graph['J'] = J
                    prop_list.append([graph['mean'],graph['std'],graph['max'],graph['min'],graph['cc'],graph['path']])
                    try:
                        if not os.path.isdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal"):
                            os.mkdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal")
                    except:
                        pass

                    file_name = "/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal/WL_graph_nn100_k{}_{}_p{:.5f}_{:07d}.p".format(node, mean, i[0],i[1])
                    name_list.append("WL_graph_nn100_k{}_{}_p{:.5f}_{:07d}.p".format(node, 100-node, i[0],i[1]))
                    with open(file_name, 'wb') as f:
                        pickle.dump(graph, f)
                        del graph

        dic['prop_list'] = prop_list
        dic['name_list'] = name_list
        with open("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_{}_{}.p".format(node, ii), 'wb') as f:
            pickle.dump(dic, f)
            del dic


# def degree(G):
#     degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
#     degree_count = collections.Counter(degree_sequence)
#     deg, cnt = zip(*degree_count.items())
#     return degree_sequence, deg, cnt
#
# new_prop_list = []
# new_name_list = []
# dic = {}
# for i in tqdm(range(20)):
#     d = pickle.load(open("/home/ubuntu/TorchGNN_project/data_temp/60_40_list_shell_{}.p".format(i), "rb"))
#     prop_list = d['prop_list']
#     name_list = d['name_list']
#     new_prop_list.append(prop_list)
#     new_name_list.append(name_list)
#
# new_name_list = np.array(new_name_list).reshape(-1,1)
# new_prop_list = np.array(new_prop_list).reshape(-1,6)
# print(new_name_list.shape, new_prop_list.shape)
# dic['prop_list'] = new_prop_list
# dic['name_list'] = new_name_list
# with open("/home/ubuntu/TorchGNN_project/data_temp/60_40_list_shell.p", "wb") as f:
#     pickle.dump(dic, f)