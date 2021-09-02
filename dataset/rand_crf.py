import os
import glob
import time

import torch
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import coo_matrix

from utils.topology import get_msg_graph
from torch.utils.data import Dataset
from utils.data_helper import *

__all__ = ['RandCRFData']

class RandCRFData(Dataset):
  def __init__(self, config, split='train'):
    # #assert split in ['train', 'val', 'test_I', 'test_II', 'test_III', 'test_IV', 'test_V', 'test_VI', 'test_VII',
    # #                 'test_VIII'], "no such split"
    # assert split in ['train', 'val', 'test',
    #                  'grid_16',
    #                  'nladder_16', 'cylinder_16', 'circladder_16', 'torrus_16',
    #                  'trilattice_16', 'hexlattice_16','nonalattice_16', 'cycle_16',
    #                  'cubic_16', 'penta_16',
    #                  'grid_9',
    #                  'nladder_8', 'cylinder_8',
    #                  'trilattice_9', 'hexlattice_10', 'nonalattice_9',
    #                  'cubic_10', 'penta_10',
    #                  'penta16_degree_8', 'penta16_degree_6', 'cubic16_degree_6',
    #                  'grid_16', 'grid_25', 'grid_36', 'grid_49', 'grid_64', 'grid_81', 'grid_100'
    #                  ], "no such split"
    self.config = config
    self.split = split
    self.data_path = config.dataset.data_path #../data_temp/
    self.data_files = sorted(glob.glob(os.path.join(self.data_path, split, '*.p')))
    self.num_graphs = len(self.data_files)
    self.npr = np.random.RandomState(seed=config.seed)

  def __getitem__(self, index):
    graph = pickle.load(open(self.data_files[index], 'rb'))
    if 'prob_gt' not in graph.keys():
      if(graph['prob_hmc'].shape[-1] != 2):
        graph['prob_gt'] = np.stack([graph['prob_hmc'], 1-graph['prob_hmc']], axis=1)
      else:
        graph['prob_gt'] = np.stack([graph['prob_hmc']], axis=1).reshape([-1, 2])

    # idx1 = self.data_files[index].find('graph_')
    # idx2 = self.data_files[index][idx1+6:].find('_')
    # graph['topology'] = self.data_files[index][idx1+6:][:idx2]
    #
    # Added by Kijung on 12/25/2019
    if 'adj' not in graph.keys():
      from utils.topology import NetworkTopology, get_msg_graph
      topology = NetworkTopology(num_nodes=len(graph['b']), seed=self.config.seed)
      # G, _ = topology.generate(topology=graph['topology'])
      G, _ = topology.generate(topology='wheel')
      graph['adj'] = topology.graph_to_adjacency_matrix(G)
      # msg_node, msg_adj = get_msg_graph(G)
      # graph['msg_node'] = msg_node
      # graph['msg_adj'] = np.asarray(msg_adj)
      graph['J'] = graph['J'].todense()
      # graph['idx_msg_edge'] = np.transpose(np.nonzero(msg_adj))

    if self.config.model.name == 'TreeReWeightedMessagePassing':
      A = graph['adj']
      graph['prob_gt'] = torch.from_numpy(graph['prob_gt']).float()
      graph['adj'] = torch.from_numpy(graph['adj']).float()
      graph['J'] = torch.from_numpy(graph['J']).float()
      graph['b'] = torch.from_numpy(graph['b']).float()

      msg_node, msg_adj = [], []
      for ii in range(self.config.model.num_trees):
        W = self.npr.rand(A.shape[0], A.shape[0])
        W = np.multiply(W, A)
        G = nx.from_numpy_matrix(W).to_undirected()
        T = nx.minimum_spanning_tree(G)
        msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
        msg_node += [msg_node_tmp]
        msg_adj += [msg_adj_tmp]

      graph['msg_node'] = torch.stack(
          [torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
      graph['msg_adj'] = torch.stack(
          [torch.from_numpy(xx.astype(float)).float() for xx in msg_adj], dim=0)
    else:
      pass
      # graph['prob_gt'] = torch.from_numpy(graph['prob_gt']).float()
      # graph['adj'] = torch.from_numpy(graph['adj']).float()
      # graph['J'] = torch.from_numpy(graph['J']).float()
      # graph['b'] = torch.from_numpy(graph['b']).float()
      # graph['msg_node'] = torch.from_numpy(np.array(graph['msg_node'])).long()
      # graph['msg_adj'] = torch.from_numpy(graph['msg_adj']).float()
    graph['file_name'] = self.data_files[index]
    return graph, 0

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch): # batch : list of dicts
    assert isinstance(batch, list)
    data = {}

    # if 'msg_node' not in batch[0].keys():
    #   data['prob_gt'] = torch.from_numpy(
    #     np.concatenate([bch['prob_gt'] for bch in batch], axis=0)).float()
    #
    #   data['b'] = torch.from_numpy(
    #     np.concatenate([bch['b'] for bch in batch], axis=0)).float()
    #
    #   n = data['b'].shape[0]
    #   data['J'] = coo_matrix(np.zeros([n, n]))
    #
    #   pad_size_l = np.array([bch['J'].shape[0] for bch in batch]).cumsum()
    #   pad_size_r = pad_size_l
    #   pad_size_r = pad_size_r[-1] - pad_size_r
    #   pad_size_l = np.concatenate(([0], pad_size_l[:-1]))
    #
    #   data['J'] = torch.from_numpy(
    #     np.stack(
    #     [
    #       np.pad(bch['J'].todense(), (pad_size_l[ii], pad_size_r[ii]), 'constant', constant_values=0.0) for ii, bch in enumerate(batch)
    #     ],
    #     axis=0).sum(axis=0)).float()
    #   G = nx.from_numpy_array(data['J'].numpy())
    #
    #   # row = []
    #   # col = []
    #   # val = []
    #   # for ii, bch in enumerate(batch):
    #   #   nv = bch['J'].shape[0]
    #   #   row.append(bch['J'].row + nv * ii)
    #   #   col.append(bch['J'].col + nv * ii)
    #   #   val.append(bch['J'].data)
    #   #
    #   # data['J'].row = np.concatenate(row)
    #   # data['J'].col = np.concatenate(col)
    #   # data['J'].data = np.concatenate(val)
    #   # G = nx.from_scipy_sparse_matrix(data['J'])
    #   #
    #   # values = torch.FloatTensor(data['J'].data)
    #   # idx = torch.LongTensor(np.vstack((data['J'].row, data['J'].col)))
    #   # data['J'] = torch.sparse.FloatTensor(idx, values, torch.Size(data['J'].shape))
    #
    #   msg_node, msg_adj = get_msg_graph(G)
    #   data['msg_node'] = torch.from_numpy(np.array(msg_node)).long()
    #   data['msg_adj'] = torch.from_numpy(np.array(msg_adj)).float()

    data['prob_gt'] = torch.from_numpy(
      np.concatenate([bch['prob_gt'] for bch in batch], axis=0)).float()
    if 'J_msg' in batch[0].keys():
      data['J_msg'] = torch.from_numpy(
        np.concatenate([bch['J_msg'] for bch in batch], axis=0)).float()
    data['b'] = torch.from_numpy(
      np.concatenate([bch['b'] for bch in batch], axis=0)).float()
    if 'bp_step' in batch[0].keys():
      data['bp_step'] = torch.from_numpy(
        np.concatenate([bch['bp_step'] for bch in batch], axis=0)).float()
    if 'x' in batch[0].keys():
      data['x'] = torch.from_numpy(
        np.concatenate([np.array([bch['x']]) for bch in batch], axis=0)).float()
    if 'y' in batch[0].keys():
      data['y'] = torch.from_numpy(
        np.concatenate([np.array([bch['y']]) for bch in batch], axis=0)).float()

    if 'msg_node' not in batch[0].keys():
      idx_msg_edge = np.empty((0, 2))
      msg_node = np.empty((0, 2))
      num_msg_node = 0
      J_m = []
      msg_adj = []
      for bch in batch:
        J = np.array(bch['J'])
        G = nx.from_numpy_array(J)
        msg_node_, msg_adj_ = get_msg_graph(G)
        msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
        idx_msg_edge_ = np.transpose(np.nonzero(msg_adj_))
        J_msg_ = J[msg_node_[:, 0], msg_node_[:, 1]].reshape(-1, 1)
        J_m.append(J_msg_)
        msg_adj.append(msg_adj_)

        idx_msg_edge = np.vstack((idx_msg_edge, msg_node.shape[0] + idx_msg_edge_))
        msg_node = np.vstack((msg_node, num_msg_node + msg_node_))
        num_msg_node = 1 + msg_node.max()

      data['J_msg'] = torch.from_numpy(
          np.concatenate(J_m, axis=0)).float()
      data['msg_adj'] = torch.from_numpy(np.array(msg_adj_, dtype=float)).long()
      data['msg_node'] = torch.from_numpy(msg_node).long()
      data['idx_msg_edge'] = torch.from_numpy(idx_msg_edge).long()

    # if 'msg_node' in batch[0].keys():
    #   idx_msg_edge = np.empty((0, 2))
    #   msg_node = np.empty((0, 2))
    #   if(len(batch[0]['msg_node'].shape) != 2):
    #     msg_node = np.empty((0, batch[0]['msg_node'].shape[1], 2))
    #   num_msg_node = 0
    #   if 'idx_msg_edge' in batch[0].keys():
    #     for bch in batch:
    #       J = np.array(bch['J'])
    #       G = nx.from_numpy_array(J)
    #       msg_node_, msg_adj_ = get_msg_graph(G)
    #       msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
    #       idx_msg_edge = np.vstack((idx_msg_edge, msg_node.shape[0] + bch['idx_msg_edge']))
    #       msg_node = np.vstack((msg_node, num_msg_node + bch['msg_node']))
    #       num_msg_node = 1 + msg_node.max()
    #   else:
    #     for bch in batch:
    #       J = np.array(bch['J'])
    #       G = nx.from_numpy_array(J)
    #       msg_node_, msg_adj_ = get_msg_graph(G)
    #       msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
    #       idx_msg_edge_ = np.transpose(np.nonzero(msg_adj_))
    #
    #       idx_msg_edge = np.vstack((idx_msg_edge, msg_node.shape[0] + idx_msg_edge_))
    #       msg_node = np.vstack((msg_node, num_msg_node + bch['msg_node']))
    #       num_msg_node = 1 + msg_node.max()
    #
    #   data['msg_adj'] = torch.from_numpy(np.array(msg_adj_).astype(np.float32)).long()
    #   data['msg_node'] = torch.from_numpy(msg_node).long()
    #   data['idx_msg_edge'] = torch.from_numpy(idx_msg_edge).long()

    idx_msg_edge = np.empty((0, 2))
    msg_node = np.empty((0, 2))
    degree = np.empty((0,1))
    if(len(batch[0]['msg_node'].shape) != 2):
      msg_node = np.empty((0, batch[0]['msg_node'].shape[1], 2))
    num_msg_node = 0
    for bch in batch:
      J = np.array(bch['J'])
      G = nx.from_numpy_array(J)
      # msg_node_, msg_adj_ = get_msg_graph(G)
      # msg_node_, msg_adj_ = np.array(msg_node_), np.array(msg_adj_)
      idx_msg_edge = np.vstack((idx_msg_edge, msg_node.shape[0] + bch['idx_msg_edge']))
      msg_node = np.vstack((msg_node, num_msg_node + bch['msg_node']))
      degree = np.vstack((degree, np.array([val for (node, val) in G.degree()]).reshape(-1,1)))
      num_msg_node = 1 + msg_node.max()

    # data['msg_adj'] = torch.from_numpy(np.array(msg_adj_).astype(np.float32)).long()
    data['msg_node'] = torch.from_numpy(msg_node).long()
    data['idx_msg_edge'] = torch.from_numpy(idx_msg_edge).long()
    data['degree'] = torch.from_numpy(degree).long()

    if 'name' in batch[0].keys() or 'file_name' in batch[0].keys():
      data['name'] = batch[0]['name']
      data['file_name'] = batch[0]['file_name']

    # values = torch.FloatTensor(batch[0]['J'].data)
    # idx = torch.LongTensor(np.vstack((batch[0]['J'].row, batch[0]['J'].col)))
    # data['J'] = torch.sparse.FloatTensor(idx, values, torch.Size(batch[0]['J'].shape))
    #

    try:
      data['J'] = torch.from_numpy(batch[0]['J']).float()
    except:
      data['J'] = batch[0]['J'].float()

    # data['msg_node'] = torch.from_numpy(batch[0]['msg_node']).long()
    # data['msg_adj'] = torch.from_numpy(batch[0]['msg_adj']).long()
    if 'msg_adj' in batch[0].keys():
      data['msg_adj'] = batch[0]['msg_adj'].long()
    return data, 0