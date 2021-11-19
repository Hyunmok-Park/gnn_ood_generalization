from utils.topology import NetworkTopology, get_msg_graph
import pickle
import os
import glob
import networkx as nx
import numpy as np
import torch.utils.data
import torch
from torch_geometric.data import Data
from utils.arg_helper import get_config
import time
from utils.data_helper import DataListLoader, DataLoader
from collections import Counter
import random

def Torchloader(config, split="train", shuffle=False, parallel=False, master_node=False, edge_module=False, sort_by_number=None, random_init=True, meta_copy=1):
    tik = time.time()
    npr = np.random.RandomState(seed=config.seed)
    data_list = []
    data_path = config.dataset.data_path
    if split == "test":
        split = config.dataset.split

    if "train" in split or "val" in split:
        batch_size = config.train.batch_size
    else:
        batch_size = config.test.batch_size
        # batch_size = 10
    print(split, batch_size)

    # data_path = "/home/ubuntu/pycharm_project_torchgnn/data_temp/grid_0.3"
    data_files = sorted(glob.glob(os.path.join(data_path, split, '*.p')))
    name_list = []
    file_list = []
    npr = np.random.RandomState(seed=0)
    if sort_by_number is None:
        for i in data_files:
            graph_data = pickle.load(open(i, "rb"))
            G = nx.from_numpy_array(graph_data['J'].todense())
            J = torch.tensor(graph_data['J'].todense()).float()
            num_nodes_I = J.size(0)
            degree = torch.tensor(np.array([val for (node, val) in G.degree()])).float()
            edge_index = torch.tensor(graph_data['msg_node']).t().contiguous().long()
            if config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                idx_msg_edge = torch.tensor(graph_data['idx_msg_edge']).t().contiguous().long()
            # edge_index[[0, 1]] = edge_index[[1, 0]]
            edge_attr = torch.tensor(graph_data['J_msg']).float()
            b = torch.tensor(graph_data['b']).float()

            if random_init:
                node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                node_idx = [[], []]
                for idx, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(idx)

                edge_idx = [[], []]
                edge_idx_inv = []
                if edge_module:
                    edge_idx_inv = [random.choice([0, 1]) for ii in range(edge_index.size(1))]
                    for idx, ii in enumerate(edge_idx_inv):
                        edge_idx[ii].append(idx)

            else:
                node_idx_inv = []
                for deg in nx.degree(G):
                    if deg[1] < 50:
                        node_idx_inv.append(0)
                    else:
                        node_idx_inv.append(1)
                node_idx = [[], []]
                for idx, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(idx)

                edge_idx = [[], []]
                edge_idx_inv = []
                if edge_module:
                    for ii in range(edge_index.size(1)):
                        if degree[edge_index[1, ii]] > 50:
                            edge_idx_inv.append(1)
                        else:
                            edge_idx_inv.append(0)

                    for idx, ii in enumerate(edge_idx_inv):
                        edge_idx[ii].append(idx)


            #################
            # D-Pattern
            #################
            # pattern_list = []
            # d = np.array([G.degree(i) for i in range(len(G.nodes()))])
            # for _ in range(len(G.nodes())):
            #     first_nei = [n for n in G.neighbors(_)]
            #     second_nei = []
            #     for nei in first_nei:
            #         for nei2 in G.neighbors(nei):
            #             second_nei.append(nei2)
            #
            #     third_nei = []
            #     for nei in second_nei:
            #         for nei2 in G.neighbors(nei):
            #             third_nei.append(nei2)
            #
            #     a = Counter(d[third_nei])
            #     pattern = np.zeros([100],dtype=int)
            #     for i in a.keys():
            #         pattern[i] = a[i]
            #
            #     pattern_list.append(pattern.tolist())
            # pattern_list = np.array(pattern_list)
            # pattern_list = torch.tensor(pattern_list)
            if "pattern_2_list" in graph_data.keys():
                pattern_list = torch.tensor(graph_data['pattern_2_list']).float()
            else:
                pattern_list = torch.zeros(1)
            #################
            # Master node
            #################
            if master_node:
                #################
                # Single Add
                #################
                # num_nodes_I = J.size(0)
                # try:
                #     std_J_A = graph_data['J_std']
                #     std_b_A = graph_data['b_std']
                # except:
                #     std_J_A = graph_data['stdJ']
                #     std_b_A = graph_data['stdb']
                # for add_num in range(1):
                #     new_edge_attr = torch.tensor(npr.normal(0, std_J_A, size=[num_nodes_I,1])).float()
                #     new_b = torch.tensor(npr.normal(0, std_b_A, size=[1, 1]))
                #     master_node_index = torch.tensor([
                #         [_ for _ in range(num_nodes_I)],
                #         [num_nodes_I for _ in range(num_nodes_I)]
                #     ])
                #     edge_index = torch.cat([edge_index, master_node_index], dim=1)
                #     master_node_index[[0, 1]] = master_node_index[[1, 0]]
                #     edge_index = torch.cat([edge_index, master_node_index], dim=1).long()
                #     edge_attr = torch.cat([edge_attr, new_edge_attr, new_edge_attr], dim=0)
                #     b = torch.cat([b, new_b], dim=0).float()
                #     num_nodes_I += 1

                #################
                # ADD COMPLETE
                #################
                num_nodes_I = J.size(0)
                topology = NetworkTopology(num_nodes=num_nodes_I, seed=config.seed)
                try:
                    std_J_A = graph_data['J_std']
                    std_b_A = graph_data['b_std']
                except:
                    std_J_A = graph_data['stdJ']
                    std_b_A = graph_data['stdb']
                G_, W_ = topology.generate(topology="complete")
                J_ = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
                J_ = (J_ + J_.transpose()) / 2.0
                J_ = J_ * W_
                new_b = torch.tensor(npr.normal(0, std_b_A, size=[num_nodes_I, 1]))
                msg_node_, msg_adj_ = get_msg_graph(G_)
                msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
                master_node_index = torch.tensor(msg_node_) + num_nodes_I

                new_edge_attr = torch.tensor(J_[msg_node_[:, 0], msg_node_[:, 1]].reshape(-1, 1)).float()
                edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
                J_ = torch.tensor(npr.normal(0, std_J_A, size=[1,1])).float()
                edge_attr = torch.cat([edge_attr, J_, J_], dim=0)

                b = torch.cat([b, new_b], dim=0).float()
                edge_index = torch.cat([edge_index, master_node_index.t()], dim=1).long()
                bridge_index = torch.tensor([
                    [num_nodes_I-1, num_nodes_I],
                    [num_nodes_I, num_nodes_I-1]
                ])
                edge_index = torch.cat([edge_index, bridge_index], dim=1).long()

            if 'prob_gt' not in graph_data.keys():
                if "prob_hmc" not in graph_data.keys():
                    y = torch.tensor(graph_data['new_gibbs']).float()
                else:
                    y = torch.tensor(graph_data['prob_hmc']).float()
            else:
                y = torch.tensor(graph_data['prob_gt']).float()

            if 'name' in graph_data.keys():
                name = graph_data['name']
            else:
                name = i

            if config.model.name == 'TreeReWeightedMessagePassing':
                topology = NetworkTopology(num_nodes=len(b), seed=config.seed)
                G, _ = topology.generate(topology='wheel')
                A = topology.graph_to_adjacency_matrix(G)

                msg_node, msg_adj = [], []
                for ii in range(config.model.num_trees):
                    W = npr.rand(A.shape[0], A.shape[0])
                    W = np.multiply(W, A)
                    G = nx.from_numpy_matrix(W).to_undirected()
                    T = nx.minimum_spanning_tree(G)
                    msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
                    msg_node += [msg_node_tmp]
                    msg_adj += [msg_adj_tmp]

                edge_index = torch.stack([torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
                msg_adj = torch.stack([torch.from_numpy(xx.astype(float)).float() for xx in msg_adj], dim=0)
                # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, msg_adj=msg_adj)
                data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, msg_adj=msg_adj)
                data_list.append(data)
                name_list.append(name)
                file_list.append(i.split('/')[-1])

            elif config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, idx_msg_edge=idx_msg_edge)
                data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, idx_msg_edge=idx_msg_edge)
            else:
                # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J)
                # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=pattern_list)#, pattern=pattern_list)
                data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, node_idx = node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)#, pattern=pattern_list)
            data_list.append(data)
            name_list.append(name)
            file_list.append(i.split('/')[-1])

    else:
        num_data = len(data_files)
        if sort_by_number == "meta_group_4":
            start_num = 16000
            second_start_num = -1
        elif sort_by_number == "meta_group_3":
            start_num = 12000
            second_start_num = -1
        elif sort_by_number == "meta_group_1":
            start_num = 4000
            second_start_num = -1
        elif sort_by_number == "meta_group_0":
            start_num = 0
            second_start_num = -1
        elif sort_by_number == "meta_group_4_0":
            start_num, second_start_num = 0, 16000
            sort_by_number1, sort_by_number2 = "meta_group_0", "meta_group_4"
        elif sort_by_number == "meta_group_3_1":
            start_num, second_start_num = 4000, 12000
            sort_by_number1, sort_by_number2 = "meta_group_1", "meta_group_3"

        if second_start_num == -1:
            for i in range(num_data):
                graph_data = pickle.load(open(os.path.join(data_path,split,"{}_{}.p".format(sort_by_number, i + start_num)), "rb"))
                G = nx.from_numpy_array(graph_data['J'].todense())
                J = torch.tensor(graph_data['J'].todense()).float()
                num_nodes_I = J.size(0)
                degree = torch.tensor(np.array([val for (node, val) in G.degree()])).float()
                edge_index = torch.tensor(graph_data['msg_node']).t().contiguous().long()
                if config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    idx_msg_edge = torch.tensor(graph_data['idx_msg_edge']).t().contiguous().long()
                # edge_index[[0, 1]] = edge_index[[1, 0]]
                edge_attr = torch.tensor(graph_data['J_msg']).float()
                b = torch.tensor(graph_data['b']).float()
                node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                node_idx = [[], []]
                for idx, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(idx)

                if random_init:
                    node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        edge_idx_inv = [random.choice([0, 1]) for ii in range(edge_index.size(1))]
                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                else:
                    node_idx_inv = []
                    for deg in nx.degree(G):
                        if deg[1] < 50:
                            node_idx_inv.append(0)
                        else:
                            node_idx_inv.append(1)
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        for ii in range(edge_index.size(1)):
                            if degree[edge_index[1, ii]] > 50:
                                edge_idx_inv.append(1)
                            else:
                                edge_idx_inv.append(0)

                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                #################
                # D-Pattern
                #################
                # pattern_list = []
                # d = np.array([G.degree(i) for i in range(len(G.nodes()))])
                # for _ in range(len(G.nodes())):
                #     first_nei = [n for n in G.neighbors(_)]
                #     second_nei = []
                #     for nei in first_nei:
                #         for nei2 in G.neighbors(nei):
                #             second_nei.append(nei2)
                #
                #     third_nei = []
                #     for nei in second_nei:
                #         for nei2 in G.neighbors(nei):
                #             third_nei.append(nei2)
                #
                #     a = Counter(d[third_nei])
                #     pattern = np.zeros([100],dtype=int)
                #     for i in a.keys():
                #         pattern[i] = a[i]
                #
                #     pattern_list.append(pattern.tolist())
                # pattern_list = np.array(pattern_list)
                # pattern_list = torch.tensor(pattern_list)
                if "pattern_2_list" in graph_data.keys():
                    pattern_list = torch.tensor(graph_data['pattern_2_list']).float()
                else:
                    pattern_list = torch.zeros(1)
                #################
                # Master node
                #################
                if master_node:
                    #################
                    # Single Add
                    #################
                    # num_nodes_I = J.size(0)
                    # try:
                    #     std_J_A = graph_data['J_std']
                    #     std_b_A = graph_data['b_std']
                    # except:
                    #     std_J_A = graph_data['stdJ']
                    #     std_b_A = graph_data['stdb']
                    # for add_num in range(1):
                    #     new_edge_attr = torch.tensor(npr.normal(0, std_J_A, size=[num_nodes_I,1])).float()
                    #     new_b = torch.tensor(npr.normal(0, std_b_A, size=[1, 1]))
                    #     master_node_index = torch.tensor([
                    #         [_ for _ in range(num_nodes_I)],
                    #         [num_nodes_I for _ in range(num_nodes_I)]
                    #     ])
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1)
                    #     master_node_index[[0, 1]] = master_node_index[[1, 0]]
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1).long()
                    #     edge_attr = torch.cat([edge_attr, new_edge_attr, new_edge_attr], dim=0)
                    #     b = torch.cat([b, new_b], dim=0).float()
                    #     num_nodes_I += 1

                    #################
                    # ADD COMPLETE
                    #################
                    num_nodes_I = J.size(0)
                    topology = NetworkTopology(num_nodes=num_nodes_I, seed=config.seed)
                    try:
                        std_J_A = graph_data['J_std']
                        std_b_A = graph_data['b_std']
                    except:
                        std_J_A = graph_data['stdJ']
                        std_b_A = graph_data['stdb']
                    G_, W_ = topology.generate(topology="complete")
                    J_ = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
                    J_ = (J_ + J_.transpose()) / 2.0
                    J_ = J_ * W_
                    new_b = torch.tensor(npr.normal(0, std_b_A, size=[num_nodes_I, 1]))
                    msg_node_, msg_adj_ = get_msg_graph(G_)
                    msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
                    master_node_index = torch.tensor(msg_node_) + num_nodes_I

                    new_edge_attr = torch.tensor(J_[msg_node_[:, 0], msg_node_[:, 1]].reshape(-1, 1)).float()
                    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
                    J_ = torch.tensor(npr.normal(0, std_J_A, size=[1,1])).float()
                    edge_attr = torch.cat([edge_attr, J_, J_], dim=0)

                    b = torch.cat([b, new_b], dim=0).float()
                    edge_index = torch.cat([edge_index, master_node_index.t()], dim=1).long()
                    bridge_index = torch.tensor([
                        [num_nodes_I-1, num_nodes_I],
                        [num_nodes_I, num_nodes_I-1]
                    ])
                    edge_index = torch.cat([edge_index, bridge_index], dim=1).long()

                if 'prob_gt' not in graph_data.keys():
                    if "prob_hmc" not in graph_data.keys():
                        y = torch.tensor(graph_data['new_gibbs']).float()
                    else:
                        y = torch.tensor(graph_data['prob_hmc']).float()
                else:
                    y = torch.tensor(graph_data['prob_gt']).float()

                if 'name' in graph_data.keys():
                    name = graph_data['name']
                else:
                    name = i

                if config.model.name == 'TreeReWeightedMessagePassing':
                    topology = NetworkTopology(num_nodes=len(b), seed=config.seed)
                    G, _ = topology.generate(topology='wheel')
                    A = topology.graph_to_adjacency_matrix(G)

                    msg_node, msg_adj = [], []
                    for ii in range(config.model.num_trees):
                        W = npr.rand(A.shape[0], A.shape[0])
                        W = np.multiply(W, A)
                        G = nx.from_numpy_matrix(W).to_undirected()
                        T = nx.minimum_spanning_tree(G)
                        msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
                        msg_node += [msg_node_tmp]
                        msg_adj += [msg_adj_tmp]

                    edge_index = torch.stack([torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
                    msg_adj = torch.stack([torch.from_numpy(xx.astype(float)).float() for xx in msg_adj], dim=0)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, msg_adj=msg_adj)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, msg_adj=msg_adj)
                    data_list.append(data)
                    name_list.append(name)


                elif config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, idx_msg_edge=idx_msg_edge)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, idx_msg_edge=idx_msg_edge)
                else:
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=pattern_list)#, pattern=pattern_list)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, node_idx = node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)#, pattern=pattern_list)
                data_list.append(data)
                name_list.append(name)
        else:
            num_data = int(num_data/2)
            for i in range(num_data):
                graph_data = pickle.load(
                    open(os.path.join(data_path, split, "{}_{}.p".format(sort_by_number1, i + start_num)), "rb"))
                G = nx.from_numpy_array(graph_data['J'].todense())
                J = torch.tensor(graph_data['J'].todense()).float()
                num_nodes_I = J.size(0)
                degree = torch.tensor(np.array([val for (node, val) in G.degree()])).float()
                edge_index = torch.tensor(graph_data['msg_node']).t().contiguous().long()
                if config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    idx_msg_edge = torch.tensor(graph_data['idx_msg_edge']).t().contiguous().long()
                # edge_index[[0, 1]] = edge_index[[1, 0]]
                edge_attr = torch.tensor(graph_data['J_msg']).float()
                b = torch.tensor(graph_data['b']).float()
                node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                node_idx = [[], []]
                for idx, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(idx)

                if random_init:
                    node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        edge_idx_inv = [random.choice([0, 1]) for ii in range(edge_index.size(1))]
                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                else:
                    node_idx_inv = []
                    for deg in nx.degree(G):
                        if deg[1] < 50:
                            node_idx_inv.append(0)
                        else:
                            node_idx_inv.append(1)
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        for ii in range(edge_index.size(1)):
                            if degree[edge_index[1, ii]] > 50:
                                edge_idx_inv.append(1)
                            else:
                                edge_idx_inv.append(0)

                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                #################
                # D-Pattern
                #################
                # pattern_list = []
                # d = np.array([G.degree(i) for i in range(len(G.nodes()))])
                # for _ in range(len(G.nodes())):
                #     first_nei = [n for n in G.neighbors(_)]
                #     second_nei = []
                #     for nei in first_nei:
                #         for nei2 in G.neighbors(nei):
                #             second_nei.append(nei2)
                #
                #     third_nei = []
                #     for nei in second_nei:
                #         for nei2 in G.neighbors(nei):
                #             third_nei.append(nei2)
                #
                #     a = Counter(d[third_nei])
                #     pattern = np.zeros([100],dtype=int)
                #     for i in a.keys():
                #         pattern[i] = a[i]
                #
                #     pattern_list.append(pattern.tolist())
                # pattern_list = np.array(pattern_list)
                # pattern_list = torch.tensor(pattern_list)
                if "pattern_2_list" in graph_data.keys():
                    pattern_list = torch.tensor(graph_data['pattern_2_list']).float()
                else:
                    pattern_list = torch.zeros(1)
                #################
                # Master node
                #################
                if master_node:
                    #################
                    # Single Add
                    #################
                    # num_nodes_I = J.size(0)
                    # try:
                    #     std_J_A = graph_data['J_std']
                    #     std_b_A = graph_data['b_std']
                    # except:
                    #     std_J_A = graph_data['stdJ']
                    #     std_b_A = graph_data['stdb']
                    # for add_num in range(1):
                    #     new_edge_attr = torch.tensor(npr.normal(0, std_J_A, size=[num_nodes_I,1])).float()
                    #     new_b = torch.tensor(npr.normal(0, std_b_A, size=[1, 1]))
                    #     master_node_index = torch.tensor([
                    #         [_ for _ in range(num_nodes_I)],
                    #         [num_nodes_I for _ in range(num_nodes_I)]
                    #     ])
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1)
                    #     master_node_index[[0, 1]] = master_node_index[[1, 0]]
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1).long()
                    #     edge_attr = torch.cat([edge_attr, new_edge_attr, new_edge_attr], dim=0)
                    #     b = torch.cat([b, new_b], dim=0).float()
                    #     num_nodes_I += 1

                    #################
                    # ADD COMPLETE
                    #################
                    num_nodes_I = J.size(0)
                    topology = NetworkTopology(num_nodes=num_nodes_I, seed=config.seed)
                    try:
                        std_J_A = graph_data['J_std']
                        std_b_A = graph_data['b_std']
                    except:
                        std_J_A = graph_data['stdJ']
                        std_b_A = graph_data['stdb']
                    G_, W_ = topology.generate(topology="complete")
                    J_ = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
                    J_ = (J_ + J_.transpose()) / 2.0
                    J_ = J_ * W_
                    new_b = torch.tensor(npr.normal(0, std_b_A, size=[num_nodes_I, 1]))
                    msg_node_, msg_adj_ = get_msg_graph(G_)
                    msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
                    master_node_index = torch.tensor(msg_node_) + num_nodes_I

                    new_edge_attr = torch.tensor(J_[msg_node_[:, 0], msg_node_[:, 1]].reshape(-1, 1)).float()
                    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
                    J_ = torch.tensor(npr.normal(0, std_J_A, size=[1, 1])).float()
                    edge_attr = torch.cat([edge_attr, J_, J_], dim=0)

                    b = torch.cat([b, new_b], dim=0).float()
                    edge_index = torch.cat([edge_index, master_node_index.t()], dim=1).long()
                    bridge_index = torch.tensor([
                        [num_nodes_I - 1, num_nodes_I],
                        [num_nodes_I, num_nodes_I - 1]
                    ])
                    edge_index = torch.cat([edge_index, bridge_index], dim=1).long()

                if 'prob_gt' not in graph_data.keys():
                    if "prob_hmc" not in graph_data.keys():
                        y = torch.tensor(graph_data['new_gibbs']).float()
                    else:
                        y = torch.tensor(graph_data['prob_hmc']).float()
                else:
                    y = torch.tensor(graph_data['prob_gt']).float()

                if 'name' in graph_data.keys():
                    name = graph_data['name']
                else:
                    name = i

                if config.model.name == 'TreeReWeightedMessagePassing':
                    topology = NetworkTopology(num_nodes=len(b), seed=config.seed)
                    G, _ = topology.generate(topology='wheel')
                    A = topology.graph_to_adjacency_matrix(G)

                    msg_node, msg_adj = [], []
                    for ii in range(config.model.num_trees):
                        W = npr.rand(A.shape[0], A.shape[0])
                        W = np.multiply(W, A)
                        G = nx.from_numpy_matrix(W).to_undirected()
                        T = nx.minimum_spanning_tree(G)
                        msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
                        msg_node += [msg_node_tmp]
                        msg_adj += [msg_adj_tmp]

                    edge_index = torch.stack([torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
                    msg_adj = torch.stack([torch.from_numpy(xx.astype(float)).float() for xx in msg_adj], dim=0)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, msg_adj=msg_adj)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, msg_adj=msg_adj)
                    data_list.append(data)
                    name_list.append(name)


                elif config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, idx_msg_edge=idx_msg_edge)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, idx_msg_edge=idx_msg_edge)
                else:
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=pattern_list)#, pattern=pattern_list)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, node_idx=node_idx,
                                node_idx_inv=node_idx_inv, edge_idx=edge_idx,
                                edge_idx_inv=edge_idx_inv)  # , pattern=pattern_list)

                for m_copy in range(meta_copy):
                    data_list.append(data)
                    name_list.append(name)

            for i in range(num_data):
                graph_data = pickle.load(
                    open(os.path.join(data_path, split, "{}_{}.p".format(sort_by_number2, i + second_start_num)), "rb"))
                G = nx.from_numpy_array(graph_data['J'].todense())
                J = torch.tensor(graph_data['J'].todense()).float()
                num_nodes_I = J.size(0)
                degree = torch.tensor(np.array([val for (node, val) in G.degree()])).float()
                edge_index = torch.tensor(graph_data['msg_node']).t().contiguous().long()
                if config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    idx_msg_edge = torch.tensor(graph_data['idx_msg_edge']).t().contiguous().long()
                # edge_index[[0, 1]] = edge_index[[1, 0]]
                edge_attr = torch.tensor(graph_data['J_msg']).float()
                b = torch.tensor(graph_data['b']).float()
                node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                node_idx = [[], []]
                for idx, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(idx)

                if random_init:
                    node_idx_inv = [random.choice([0, 1]) for ii in range(num_nodes_I)]
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        edge_idx_inv = [random.choice([0, 1]) for ii in range(edge_index.size(1))]
                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                else:
                    node_idx_inv = []
                    for deg in nx.degree(G):
                        if deg[1] < 50:
                            node_idx_inv.append(0)
                        else:
                            node_idx_inv.append(1)
                    node_idx = [[], []]
                    for idx, ii in enumerate(node_idx_inv):
                        node_idx[ii].append(idx)

                    edge_idx = [[], []]
                    edge_idx_inv = []
                    if edge_module:
                        for ii in range(edge_index.size(1)):
                            if degree[edge_index[1, ii]] > 50:
                                edge_idx_inv.append(1)
                            else:
                                edge_idx_inv.append(0)

                        for idx, ii in enumerate(edge_idx_inv):
                            edge_idx[ii].append(idx)

                #################
                # D-Pattern
                #################
                # pattern_list = []
                # d = np.array([G.degree(i) for i in range(len(G.nodes()))])
                # for _ in range(len(G.nodes())):
                #     first_nei = [n for n in G.neighbors(_)]
                #     second_nei = []
                #     for nei in first_nei:
                #         for nei2 in G.neighbors(nei):
                #             second_nei.append(nei2)
                #
                #     third_nei = []
                #     for nei in second_nei:
                #         for nei2 in G.neighbors(nei):
                #             third_nei.append(nei2)
                #
                #     a = Counter(d[third_nei])
                #     pattern = np.zeros([100],dtype=int)
                #     for i in a.keys():
                #         pattern[i] = a[i]
                #
                #     pattern_list.append(pattern.tolist())
                # pattern_list = np.array(pattern_list)
                # pattern_list = torch.tensor(pattern_list)
                if "pattern_2_list" in graph_data.keys():
                    pattern_list = torch.tensor(graph_data['pattern_2_list']).float()
                else:
                    pattern_list = torch.zeros(1)
                #################
                # Master node
                #################
                if master_node:
                    #################
                    # Single Add
                    #################
                    # num_nodes_I = J.size(0)
                    # try:
                    #     std_J_A = graph_data['J_std']
                    #     std_b_A = graph_data['b_std']
                    # except:
                    #     std_J_A = graph_data['stdJ']
                    #     std_b_A = graph_data['stdb']
                    # for add_num in range(1):
                    #     new_edge_attr = torch.tensor(npr.normal(0, std_J_A, size=[num_nodes_I,1])).float()
                    #     new_b = torch.tensor(npr.normal(0, std_b_A, size=[1, 1]))
                    #     master_node_index = torch.tensor([
                    #         [_ for _ in range(num_nodes_I)],
                    #         [num_nodes_I for _ in range(num_nodes_I)]
                    #     ])
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1)
                    #     master_node_index[[0, 1]] = master_node_index[[1, 0]]
                    #     edge_index = torch.cat([edge_index, master_node_index], dim=1).long()
                    #     edge_attr = torch.cat([edge_attr, new_edge_attr, new_edge_attr], dim=0)
                    #     b = torch.cat([b, new_b], dim=0).float()
                    #     num_nodes_I += 1

                    #################
                    # ADD COMPLETE
                    #################
                    num_nodes_I = J.size(0)
                    topology = NetworkTopology(num_nodes=num_nodes_I, seed=config.seed)
                    try:
                        std_J_A = graph_data['J_std']
                        std_b_A = graph_data['b_std']
                    except:
                        std_J_A = graph_data['stdJ']
                        std_b_A = graph_data['stdb']
                    G_, W_ = topology.generate(topology="complete")
                    J_ = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
                    J_ = (J_ + J_.transpose()) / 2.0
                    J_ = J_ * W_
                    new_b = torch.tensor(npr.normal(0, std_b_A, size=[num_nodes_I, 1]))
                    msg_node_, msg_adj_ = get_msg_graph(G_)
                    msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
                    master_node_index = torch.tensor(msg_node_) + num_nodes_I

                    new_edge_attr = torch.tensor(J_[msg_node_[:, 0], msg_node_[:, 1]].reshape(-1, 1)).float()
                    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
                    J_ = torch.tensor(npr.normal(0, std_J_A, size=[1, 1])).float()
                    edge_attr = torch.cat([edge_attr, J_, J_], dim=0)

                    b = torch.cat([b, new_b], dim=0).float()
                    edge_index = torch.cat([edge_index, master_node_index.t()], dim=1).long()
                    bridge_index = torch.tensor([
                        [num_nodes_I - 1, num_nodes_I],
                        [num_nodes_I, num_nodes_I - 1]
                    ])
                    edge_index = torch.cat([edge_index, bridge_index], dim=1).long()

                if 'prob_gt' not in graph_data.keys():
                    if "prob_hmc" not in graph_data.keys():
                        y = torch.tensor(graph_data['new_gibbs']).float()
                    else:
                        y = torch.tensor(graph_data['prob_hmc']).float()
                else:
                    y = torch.tensor(graph_data['prob_gt']).float()

                if 'name' in graph_data.keys():
                    name = graph_data['name']
                else:
                    name = i

                if config.model.name == 'TreeReWeightedMessagePassing':
                    topology = NetworkTopology(num_nodes=len(b), seed=config.seed)
                    G, _ = topology.generate(topology='wheel')
                    A = topology.graph_to_adjacency_matrix(G)

                    msg_node, msg_adj = [], []
                    for ii in range(config.model.num_trees):
                        W = npr.rand(A.shape[0], A.shape[0])
                        W = np.multiply(W, A)
                        G = nx.from_numpy_matrix(W).to_undirected()
                        T = nx.minimum_spanning_tree(G)
                        msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
                        msg_node += [msg_node_tmp]
                        msg_adj += [msg_adj_tmp]

                    edge_index = torch.stack([torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
                    msg_adj = torch.stack([torch.from_numpy(xx.astype(float)).float() for xx in msg_adj], dim=0)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, msg_adj=msg_adj)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, msg_adj=msg_adj)
                    data_list.append(data)
                    name_list.append(name)


                elif config.model.name == "TorchGNN_MsgGNN" or config.model.name == "TorchGNN_MsgGNN_parallel":
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J, idx_msg_edge=idx_msg_edge)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, idx_msg_edge=idx_msg_edge)
                else:
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=degree, J=J)
                    # data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, degree=pattern_list)#, pattern=pattern_list)
                    data = Data(x=b, edge_index=edge_index, edge_attr=edge_attr, y=y, J=J, node_idx=node_idx,
                                node_idx_inv=node_idx_inv, edge_idx=edge_idx,
                                edge_idx_inv=edge_idx_inv)  # , pattern=pattern_list)

                for m_copy in range(meta_copy):
                    data_list.append(data)
                    name_list.append(name)

    if parallel:
        loader = DataListLoader(data_list, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    else:
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    print("data loading time: ", time.time() - tik)
    return loader, name_list, file_list


if __name__ == '__main__':
    # config = yaml.load(open("config/node_gnn10.yaml", 'r'), Loader=yaml.FullLoader)
    config = get_config("config/node_gnn10.yaml", sample_id="{:03d}".format(0))
    tik = time.time()
    train_loader, _ = Torchloader(config, split='train')
    print(time.time() - tik)
    for i in train_loader:
        print(i)
        print(asdf)
