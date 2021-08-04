from sklearn.decomposition import PCA
import pickle
import collections
import numpy as np
import os
import shutil
from tqdm import tqdm
import networkx as nx
import argparse
# @func_set_timeout(60)
def iso(G1, G2):
    return nx.is_isomorphic(G1, G2)

def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt

def find_graph_name(index):
    data = pickle.load(open("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell_UNION.p", "rb"))
    name_list = data['name_list']
    data = data['prop_list']
    print("data_loaded")

    data_mean = np.mean(data, axis=0)
    data_scaled = data-data_mean
    pca = PCA(n_components=6)
    pca.fit(data_scaled)
    X_low = pca.transform(data_scaled)

    x6 = []
    x7 = []
    x8 = []
    x9 = []
    for x, i in enumerate(X_low):
        if -46.75 < i[0] < -46.25 and 9.5 < -i[1] < 9.75: #3.5
            x6.append(x)
        if -13 < i[0] < -12.5 and 9.75 < -i[1] < 10: #2.5
            x7.append(x)
        if 19.75 < i[0] < 20.50 and 10.75 < -i[1] < 11: #1.5
            x8.append(x)
        if 54.5 < i[0] < 55 and 11.75 < -i[1] < 12: #0.5
            x9.append(x)
    if index == 0:
        b = name_list[x9]
    elif index == 1:
        b = name_list[x8]
    elif index == 2:
        b = name_list[x7]
    elif index == 3:
        b = name_list[x6]
    return b

def main(index):
    name_list = find_graph_name(index)
    J_data = []
    name = []
    for i in tqdm(name_list):
        graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}.5".format(index), i)
        graph = pickle.load(open(graph_path, "rb"))
        J = graph['J']
        J_data.append(J)
        name.append(i)

    G = nx.from_numpy_array(J.todense())
    _, deg, _ = degree(G)

    save_dir = "/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}.5_2".format(index)
    print(len(J_data), len(name))
    idx = [i for i in range(len(J_data))]
    iso_list = []
    try:
        os.mkdir(save_dir)
    except:
        pass

    count = 0
    while True:
        J1 = J_data[idx[0]].todense()
        G1 = nx.from_numpy_array(J1)
        iso_list.append(name[idx[0]])
        remove_list = [idx[0]]
        src = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}.5".format(index), name[idx[0]])
        dst = os.path.join(save_dir, name[idx[0]])
        shutil.copyfile(src, dst)

        for i in tqdm(idx[1:], leave=False):
            J2 = J_data[i].todense()
            G2 = nx.from_numpy_array(J2)

            try:
                a = iso(G1, G2)
            except:
                try:
                    a = iso(G2, G1)
                except:
                    pass
            # print(idx[0], i)
            if a:
                remove_list.append(i)

        for i in remove_list:
            idx.remove(i)

        if len(idx) <= 1:
            if len(idx)==1:
                iso_list.append(name[idx[0]])
            break
        print("current iso len :", len(iso_list))
        print("current remain len :", len(idx))

        count += 1
        if count > 500:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=1, help='index')
    args = parser.parse_args()

    main(args.index)