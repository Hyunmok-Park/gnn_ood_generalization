import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
import collections
from collections import Counter
from tqdm import tqdm
import shutil

def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt


def func(x, a, b, c):
    return a * (x - b) ** 2 + c

# name = pickle.load(open(os.path.join('GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-53_train_100_group_4_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-50_exp2_test_100_0.3_add_marginal33_64_10_add______', "name.p"), "rb"))
# deg_list = []
# for graph in tqdm(name):
#     graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/asdf", graph)
#     G = pickle.load(open(graph_path, "rb"))
#     J = G['J'].todense()
#     G = nx.from_numpy_array(J)
#     _, deg, _ = degree(G)
#     deg_list.append(deg)
#
# with open("deg_list.p", "wb") as f:
#     pickle.dump(deg_list, f)


for i in range(5):
    deg_list = []
    for ii in tqdm(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}".format(i))):
        graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}".format(i), ii)
        graph = pickle.load(open(graph_path, "rb"))
        J = graph['J'].todense()
        G = nx.from_numpy_array(J)
        _, deg, _ = degree(G)
        deg_list.append(deg)
    with open("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}.p".format(i), "wb") as f:
        pickle.dump(deg_list, f)

# for i in os.listdir("/home/ubuntu/TorchGNN_project/data_temp/add_data"):
#     graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/add_data", i)
#     graph = pickle.load(open(graph_path, "rb"))
#     J = graph['J'].todense()
#     G = nx.from_numpy_array(J)
#     _, deg, _ = degree(G)
#     print(deg)

# test_path = [
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-53_train_100_group_4_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-50_exp2_test_100_0.3_add_marginal33_64_10_add______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-42_train_100_group_4_64_10_add__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-09-48_exp2_test_100_0.3_add_marginal33_64_10_add__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-45_train_100_group_4_64_10_add__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-09-49_exp2_test_100_0.3_add_marginal33_64_10_add__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-28_train_100_group_4_64_10_att___/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-22-19_exp2_test_100_0.3_add_marginal33_64_10_att______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-48_train_100_group_4_64_10_att__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-09-50_exp2_test_100_0.3_add_marginal33_64_10_att__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-51_train_100_group_4_64_10_att__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-09-51_exp2_test_100_0.3_add_marginal33_64_10_att__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-49_train_100_group_3_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-48_exp2_test_100_0.3_add_marginal33_64_10_add______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-18_train_100_group_3_64_10_add__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-52-50_exp2_test_100_0.3_add_marginal33_64_10_add__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-21_train_100_group_3_64_10_add__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-55-32_exp2_test_100_0.3_add_marginal33_64_10_add__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-31_train_100_group_3_64_10_att___/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-22-20_exp2_test_100_0.3_add_marginal33_64_10_att______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-24_train_100_group_3_64_10_att__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-01-21_exp2_test_100_0.3_add_marginal33_64_10_att__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-27_train_100_group_3_64_10_att__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-01-22_exp2_test_100_0.3_add_marginal33_64_10_att__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-47_train_100_group_2_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-48_exp2_test_100_0.3_add_marginal33_64_10_add______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-06_train_100_group_2_64_10_add__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-15-57-50_exp2_test_100_0.3_add_marginal33_64_10_add__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-09_train_100_group_2_64_10_add__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-50-08_exp2_test_100_0.3_add_marginal33_64_10_add__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-34_train_100_group_2_64_10_att___/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-22-21_exp2_test_100_0.3_add_marginal33_64_10_att______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-12_train_100_group_2_64_10_att__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-42-26_exp2_test_100_0.3_add_marginal33_64_10_att__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-15_train_100_group_2_64_10_att__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-42-27_exp2_test_100_0.3_add_marginal33_64_10_att__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-43_train_100_group_1_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-47_exp2_test_100_0.3_add_marginal33_64_10_add______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-07-54_train_100_group_1_64_10_add__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-15-57-45_exp2_test_100_0.3_add_marginal33_64_10_add__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-07-57_train_100_group_1_64_10_add__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-15-57-46_exp2_test_100_0.3_add_marginal33_64_10_add__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-37_train_100_group_1_64_10_att___/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-22-22_exp2_test_100_0.3_add_marginal33_64_10_att______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-00_train_100_group_1_64_10_att__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-15-57-47_exp2_test_100_0.3_add_marginal33_64_10_att__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-03_train_100_group_1_64_10_att__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-15-57-48_exp2_test_100_0.3_add_marginal33_64_10_att__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-40_train_100_group_0_64_10_add/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-18-46_exp2_test_100_0.3_add_marginal33_64_10_add______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-30_train_100_group_0_64_10_add__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-01-23_exp2_test_100_0.3_add_marginal33_64_10_add__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-33_train_100_group_0_64_10_add__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-01-24_exp2_test_100_0.3_add_marginal33_64_10_add__SK_IP___',
# 'GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-18-18-56-21_train_100_group_0_64_10_att___/TorchGNN_001_Torchloader_data_temp_2021-May-13-16-22-23_exp2_test_100_0.3_add_marginal33_64_10_att______',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-36_train_100_group_0_64_10_att__SK_/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-01-25_exp2_test_100_0.3_add_marginal33_64_10_att__SK____',
# 'GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-39_train_100_group_0_64_10_att__SK_IP/TorchGNN_001_TorchGeoLoader_data_temp_2021-May-13-16-09-46_exp2_test_100_0.3_add_marginal33_64_10_att__SK_IP___']
#
#
#
#
# number_of_model = 6
# index = 1
# f = plt.figure(figsize=(28, 27))
# for x, path in tqdm(enumerate(test_path), desc = "OUT"):
#     XYL = np.array(pickle.load(open(os.path.join(path, "XYL.p"), "rb")))
#     print(XYL.shape)
#     min_L = np.min(XYL[:, 2])
#     max_L = np.max(XYL[:, 2])
#
#     threshold = 0.1 * (min_L + max_L)
#     graph_index_toplot = np.where(XYL[:, 2] < threshold)[0]
#     print("NUMBER OF GRAPH : ", len(graph_index_toplot))
#     name = pickle.load(open(os.path.join(path, "name.p"), "rb"))
#     graph_name = name[graph_index_toplot]
#
#     deg_list = []
#     ax = f.add_subplot(11, 13, int(x / number_of_model) + (x % number_of_model) * 13 + 1)
#     for graph in tqdm(graph_name, desc = "IN"):
#         graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/asdf", graph)
#         G = pickle.load(open(graph_path, "rb"))
#         J = G['J'].todense()
#         G = nx.from_numpy_array(J)
#         _, deg, _ = degree(G)
#         deg_list += deg
#
#     a = Counter(deg_list)
#     sum_count = sum(a.values())
#     plt.bar(a.keys(), [i / sum_count for i in list(a.values())], fill=True, edgecolor='blue', width=0.4,
#     label = 'Test')
#     plt.ylabel("Distribution")
#     plt.xlabel("Degree")
#
#     if x in [0,6,12,18,24]:
#         train_deg_list = []
#         train_group = 4 - int(x / number_of_model)
#         for graph in tqdm(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}".format(train_group)), desc="Train"):
#             graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_{}".format(train_group), graph)
#             G = pickle.load(open(graph_path, "rb"))
#             J = G['J'].todense()
#             G = nx.from_numpy_array(J)
#             _, deg, _ = degree(G)
#             train_deg_list += deg
#
#     a = Counter(train_deg_list)
#     sum_count = sum(a.values())
#     plt.bar(a.keys(), [i / sum_count for i in list(a.values())], fill=False, edgecolor='orange', width=0.4,
#             label='Train')
#
#     plt.legend()
#     plt.xlim(0, 100)
#
#     plt.subplots_adjust(hspace=0.3)
# plt.savefig("test.pdf", bbox_inches='tight')
