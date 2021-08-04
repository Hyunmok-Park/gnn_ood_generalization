import os
import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.gt_inference import Enumerate
from scipy.sparse import coo_matrix
import argparse
from multiprocessing import Process
from model.bp import BeliefPropagation
import networkx as nx
from itertools import product
from tqdm import tqdm

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main(index):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
  """
  std_J_A = 0.3
  std_b_A = 0.25
  num_nodes_I = 100
  save_dir = "/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_80_SP_JB"

  try:
      mkdir(save_dir)
  except OSError:
      pass

  s = [
    "/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_80_SP"
  ]
  len_index = int(len(os.listdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_80_SP")) / 20) + 1
  graph_dir = s[0]
  for sample_id, graph_name in zip([i for i in range(len_index*index,len_index*(index+1))], os.listdir(graph_dir)[len_index*index:len_index*(index+1)]):
      # save_dir = '../data_temp_server/unique_degree/{}'.format(Unique_degree)
      #npr = np.random.RandomState(seed=seed_train)

      #############################################################################
      # Generate Training Graphs
      #############################################################################

      # here seed only affects random graphs
      print('Generating training graphs!')
      #topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)

      graph_path = os.path.join(graph_dir, graph_name)
      graph_dict = pickle.load(open(graph_path, "rb"))

      G, W = nx.from_numpy_array(graph_dict["J"].todense()), graph_dict["J"].todense()

      for i,j in product(range(100), range(100)):
          if(W[i,j] != 0):
              W[i, j] = 1

      seed_test = int(str(3333) + str(sample_id))#
      npr = np.random.RandomState(seed=seed_test)#
      graph = {}

      J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
      J = (J + J.transpose()) / 2.0
      J = J * np.array(W)
      b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

      # Enumerate
      # model = Enumerate(W, J, b)
      # prob_gt = model.inference()
      # map_gt = model.inference_map()

      # graph['prob_gt'] = prob_gt # shape N x 2
      # graph['map_gt'] = map_gt  # shape N x 2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_test'] = seed_test
      graph['stdJ'] = std_J_A
      graph['name'] = graph_name

      msg_node, msg_adj = get_msg_graph(G)
      msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
      idx_msg_edge = np.transpose(np.nonzero(msg_adj))
      J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

      graph['msg_node'] = msg_node
      graph['idx_msg_edge'] = idx_msg_edge
      graph['J_msg'] = J_msg

      # BeliefPropagation
      # bp_prob, bp_step = BeliefPropagation(J, b, msg_node, msg_adj).inference(max_iter = 15, damping = 0.0, observe_mask=None)
      # graph['bp_step'] = bp_step

      file_name = os.path.join(save_dir, 'graph_WL_{}_{:07d}.p'.format(graph_name, sample_id))
      sample_id+=1
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)
        del graph


# python -m dataset.gen_train2
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=1, help='index')
    args = parser.parse_args()

    main(args.index)