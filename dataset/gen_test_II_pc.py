import os
import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.block_gibbs import BlockGibbs
from model.hmc import HMC
from model.min_cut import MinCut
from scipy.sparse import coo_matrix
import argparse
from multiprocessing import Process
import time
from tqdm import tqdm
import networkx as nx


def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))


def main(begin, end):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
    Test Protocols:
      I:  |V| = 9, 13 special structures, random potentials A
      II: |V| = 100, 13 special structures, random potentials A
  """
  #seed_test = int(str(3333)+str(sample_id))
  std_J_A = 0.9
  std_b_A = 0.25
  num_nodes_II = 100
  num_graphs_test = 10
  save_dir = '/home/ubuntu/data_temp'

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  block_size = 4

  hmc_burn_in = 10000
  hmc_num_sample = 10000
  trv_time = 49.5

  print('Generating training graphs!')

  try:
    mkdir(save_dir)  #!!
  except OSError:
    pass

  path = "/home/ubuntu/data_temp/WL_flex_graphs_100_sub_path5"
  # graph_list = Graphs[begin, end]

  for ii in tqdm(range(begin, end)):
      # graph_path = os.path.join(path, Graphs[ii])
      graph_path = "/home/ubuntu/data_temp/WL_flex_graphs_100_pca_sub_50/WL_graph_nn100_k98.0_p0.00000_0000000.p"
      _ = pickle.load(open(graph_path, "rb"))
      G, W = nx.from_numpy_array(_['J'].todense()), _['J'].todense()

      sample_id = ii  #
      seed_test = int(str(3333) + str(sample_id))  #

      npr = np.random.RandomState(seed=seed_test)  #
      graph = {}
      J = npr.normal(0, std_J_A, size=[num_nodes_II, num_nodes_II])
      J = (J + J.transpose()) / 2.0
      J = J * np.array(W)
      b = npr.normal(0, std_b_A, size=[num_nodes_II, 1])

      # Block Gibbs Sampling
      tik1 = time.time()
      model = BlockGibbs(G, J, b, block_method='gibbs', block_size=block_size, seed=seed_test)
      prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
      gibbs_time = time.time() - tik1

      # Hamiltonian Monte Carlo
      tik2 = time.time()
      model = HMC(W, J, b, seed=seed_test)
      prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)
      hmc_time = time.time() - tik2

      graph['prob_gibbs'] = prob_gibbs  # shape N x 2
      graph['prob_hmc'] = prob_hmc
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_test'] = seed_test
      graph['J_std'] = std_J_A

      msg_node, msg_adj = get_msg_graph(G)
      msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
      idx_msg_edge = np.transpose(np.nonzero(msg_adj))
      J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

      graph['msg_node'] = msg_node
      graph['idx_msg_edge'] = idx_msg_edge
      graph['J_msg'] = J_msg

      graph['gibb_burn_in'] = 10000
      graph['gibb_num_sample'] = 10000
      graph['block_size'] = 4

      graph['hmc_burn_in'] = 10000
      graph['hmc_num_sample'] = 10000
      graph['trv_time']= 49.5

      graph['gibbs_time'] = gibbs_time
      graph['hmc_time'] = hmc_time

      file_name = os.path.join(save_dir, 'graph_WL_nn{}_{:07d}.p'.format(num_nodes_II, sample_id))
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)
        del graph


# python3 -m dataset.gen_test_II_pc
# Use different seed for train/val/test
if __name__ == '__main__':
    jobs = []
    Graphs = sorted(os.listdir("/home/ubuntu/data_temp/WL_flex_graphs_100_pca_sub_50"))

    for i in range(16):
        if (i == 15):
            begin, end = 0,1
        else:
            begin, end = 0,1

        p = Process(target=main, args=(begin, end))
        jobs.append(p)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()
