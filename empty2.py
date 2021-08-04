from __future__ import (division, print_function)

import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import collections
from multiprocessing import Manager, Process, Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shutil
from dataset import *
import argparse
import pickle
import numpy as np
import networkx as nx

def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt


# def main(a, index):
#     l1 = []
#     l2 = []
#
#     for i in tqdm(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell")[180000*index:180000*(index+1)]):
#         G = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell", i)
#         G = pickle.load(open(G, "rb"))
#         J = G['J'].todense()
#         G = nx.from_numpy_array(J)
#         deg1, deg2, _ = degree(G)
#
#         l1.append([np.mean(deg1), np.std(deg1)])
#         l2.append([np.mean(deg2), np.std(deg2)])
#
#     l1 = np.array(l1)
#     l2 = np.array(l2)
#     d = {}
#
#     d['unique'] = l2
#     d['non_unique'] = l1
#     with open("uni_vs_nonuni_{}.p".format(index), "wb") as f:
#         pickle.dump(d, f)


    # path = "/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell_UNION"
    #
    # prop_list = []
    # name_list = []
    # dict = {}
    # ll = int(int(len(os.listdir(path))) / 20)
    # print(ll)
    #
    # for i in tqdm(sorted(os.listdir(path))[index*ll:(index+1)*ll]):
    #     G = os.path.join(path, i)
    #     G = pickle.load(open(G, "rb"))
    #     J = G['J'].todense()
    #     G = nx.from_numpy_array(J)
    #     deg, _, _ = degree(G)
    #
    #     prop = [G['mean'], G['std'], G['max'], G['min'], G['cc'], G['path']]
    #
    #     name_list.append(i)
    #     prop_list.append(prop)
    #
    # prop_list = np.array(prop_list)
    # dict['name_list'] = name_list
    # dict['prop_list'] = prop_list
    #
    # with open("/home/ubuntu/TorchGNN_project/data_temp/100_list_shell_UNION_not_unique_{}.p".format(index), "wb") as f:
    #     pickle.dump(dict, f)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--index', type=str, default=0, help='index')
#     args = parser.parse_args()
#
#     p = Pool(processes=20)
#     p.starmap(main, zip([i for i in range(20)],[i for i in range(20)]))


# path1 = "/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4/train"
# for i in tqdm(os.listdir(path1)):
#     s = os.path.join(path1, i)
#     d = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4_2/train", i)
#     shutil.copyfile(s,d)
#
# path2 = "/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4/val"
# for i in os.listdir(path2):
#     s = os.path.join(path2, i)
#     d = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4_2/val", i)
#     shutil.copyfile(s,d)
#
# path3 = "/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_2/train"
# for i in tqdm(os.listdir(path3)):
#     s = os.path.join(path3, i)
#     d = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4_2/train", i)
#     shutil.copyfile(s,d)
#
# path4 = "/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_2/val"
# for i in os.listdir(path4):
#     s = os.path.join(path4, i)
#     d = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4_2/val", i)
#     shutil.copyfile(s,d)

# d = '/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal/WL_graph_nn100_k3_97_p0.00000_0000000.p'
# pickle.load(open(d, "rb"))

def main(a,b):
    name_list = []
    prop_list = []
    total_file = len(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal"))
    tik = math.ceil(total_file / 20)
    for i in tqdm(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal")[a*tik:(a+1)*tik]):
        graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal", i)
        graph = pickle.load(open(graph_path, "rb"))
        name_list.append(i)
        prop_list.append([graph['mean'], graph['std'], graph['max'], graph['min'], graph['cc'], graph['path']])

    dd = {}
    dd['name_list'] = name_list
    dd['prop_list'] = prop_list
    with open("WS_flex_graph_100_bimodal_{}.p".format(a), "wb") as f:
        pickle.dump(dd, f)

p = Pool(processes=20)
p.starmap(main, zip([i for i in range(20)],[i for i in range(20)]))

