import argparse
import collections
from itertools import product
from networkx.utils import py_random_state
import networkx as nx
from tqdm import tqdm
import numpy as np
import pickle
import os

def compute_count(channel, group):
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out


@py_random_state(3)
def ws_graph(n, k, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """
    assert k >= 2 and k <= n
    # compute number of edges:
    edge_num = int(round(k * n / 2))
    count = compute_count(edge_num, n)
    # print(count)
    G = nx.Graph()
    for i in tqdm(range(n), leave=False):
        source = [i] * count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
        # print(source, target)
        G.add_edges_from(zip(source, target))
    # rewire edges from each node
    nodes = list(G.nodes())
    for i in tqdm(range(n), leave=False):
        u = i
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
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
    return G

@py_random_state(4)
def connected_ws_graph(n, k, p, tries=1000, seed=1):
    """Returns a connected ws-flex graph.
    """
    for i in tqdm(range(tries), desc='try', leave=False):
        # print("try", i)
        # seed is an RNG so should change sequence each call
        G = ws_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    print("max try: k{}_p{}_seed{}".format(k,p,seed))

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


    for node in [10, 20, 30, 40]:
        # np.random.seed(node)
        # node2 = np.random.choice([i for i in range(10,21)])
        # node3 = 102 - node - node2
        # node2 = node3 = node
        # print(node, node2, node3)

        degree_space = np.linspace(2, node, 300)
        prob_space = np.linspace(0, 1, 300)**2
        seed_space = [i+50 for i in range(40)]

        space = list(product(degree_space, prob_space, seed_space))
        ii = args.index
        prop_list = []
        name_list = []
        dic = {}

        for i in space[225000 * ii:225000 * (ii + 1)]:
            graph = {}
            G = connected_ws_graph(node, i[0], i[1], 500, i[2])

            #Union of WS_random_graph
            # number_to_combine = (node * 3 - 100) / 2
            # index_increase = node - number_to_combine
            # G2 = connected_ws_graph(node2, np.random.choice([i for i in range(2, node2)]), i[1], 500, i[2])
            # # mapping = dict(zip(G2, [i + node - 1 for i in range(node2)]))
            # mapping = dict(zip(G2, [i + index_increase for i in range(node2)]))
            # G2 = nx.relabel_nodes(G2, mapping)
            # G = nx.compose(G, G2)
            # print(len(nx.nodes(G)))
            #
            # G3 = connected_ws_graph(node3, np.random.choice([i for i in range(2, node3)]), i[1], 500, i[2])
            # # mapping = dict(zip(G3, [i + (node+node2) - 2 for i in range(node3)]))
            # mapping = dict(zip(G3, [i + index_increase*2 for i in range(node3)]))
            # G3 = nx.relabel_nodes(G3, mapping)
            # G = nx.compose(G, G3)
            # print(len(nx.nodes(G)))
            #
            assert(len(nx.nodes(G)) == 100)

            #Connect 100-|V| WS_random_graph
            # node2 = 100-node+1
            # G2 = connected_ws_graph(node2, np.random.choice([i for i in range(2, 100-node+1)]), i[1], 500, i[2])
            # mapping = dict(zip(G2, [i + node -1 for i in range(100-node+1)]))
            # G2 = nx.relabel_nodes(G2, mapping)
            # G = nx.compose(G, G2)

            #Connect path graph
            # for _ in range(100-node):
            #     G.add_edge(node - 1 + _, node + _)

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
                    if not os.path.isdir("/home/ubuntu/TorchGNN_project/data_temp/WS_TEST"):
                        os.mkdir("/home/ubuntu/TorchGNN_project/data_temp/WS_TEST")
                except:
                    pass

                file_name = "/home/ubuntu/TorchGNN_project/data_temp/WS_TEST/WL_graph_nn{}_k{}_p{:.5f}_{:07d}.p".format(node, i[0],i[1],i[2])
                name_list.append("WL_graph_nn{}_k{}_p{:.5f}_{:07d}.p".format(node, i[0],i[1],i[2]))
                with open(file_name, 'wb') as f:
                    pickle.dump(graph, f)
                    del graph

        dic['prop_list'] = prop_list
        dic['name_list'] = name_list
        with open("/home/ubuntu/TorchGNN_project/data_temp/WS_TEST_{}.p".format(ii), 'wb') as f:
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