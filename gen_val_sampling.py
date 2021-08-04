import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
# from hmc import HMC
# from block_gibbs import BlockGibbs
from multiprocessing import Manager, Process, Pool
from scipy.sparse import coo_matrix
import argparse
import math
# import time
# from tqdm import tqdm
import networkx as nx
# from itertools import repeat
# import shutil
# from gt_inference import Enumerate
# from itertools import product

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))


def main(index, try_num, group_num):
# def main(group_index, index, split_num, tt_list, name_list, seed_list):
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
  # assert index < split_num
  print("index:", index)
  num_nodes_I = 100
  std_J_A = 0.3
  std_b_A = 0.25
  num_graphs_test = 100

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  block_size = 4

  hmc_burn_in = 10000
  hmc_num_sample = 10000
  trv_time = 49.5

  print('Generating training graphs!')

  save_dir = 'data_temp/exp2_train_100_0.3/group_two_unimodal'
  J_list = pickle.load(open("data_temp/unique_graph_100/group_two_unimodal_val_{}.p".format(try_num), "rb"))['J_list']
  name_list_full = pickle.load(open("data_temp/unique_graph_100/group_two_unimodal_val_{}.p".format(try_num), "rb"))['name_list']

  n = math.ceil(len(J_list)/ 20)

  tt_list = J_list[n * index:n * (index+1)]
  name_list = name_list_full[n * index:n * (index+1)]
  seed_list = [i+2000*try_num for i in range(2000)][n * index:n * (index+1)]

  print("total length", len(J_list), len(tt_list))
  del J_list, name_list_full

  for tt, seed, name in zip(tt_list, seed_list, name_list):
      seed_val = int(str(2222) + str(seed))
      npr = np.random.RandomState(seed=seed_val)

      # print(tt)
      #############################################################################
      # Generate Training Graphs
      #############################################################################
      # here seed only affects random graphs
      print('Generating training graphs!')
      # topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)

      try:
          mkdir(os.path.join(save_dir, 'val'))
          # mkdir(os.path.join(save_dir, 'val', 'pass'))
          # mkdir(os.path.join(save_dir, 'val', 'fail'))
      except OSError:
          pass

      graph = {}

      try:
        W = tt.todense()
      except:
        W = tt

      G = nx.from_numpy_array(W)
      J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
      J = (J + J.transpose()) / 2.0
      J = J * np.array(W)
      b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

      # # Block Gibbs Sampling
      # tik1 = time.time()
      # model = BlockGibbs(G, J, b, block_method='gibbs', block_size=block_size, seed=seed_val)
      # prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
      # gibbs_time = time.time() - tik1
      #
      # # Hamiltonian Monte Carlo
      # tik2 = time.time()
      # model = HMC(W, J, b, seed=seed_val)
      # prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)
      # hmc_time = time.time() - tik2
      #
      # graph['prob_hmc'] = prob_hmc
      # graph['prob_gibbs'] = prob_gibbs

      #Enumerate
      # model = Enumerate(W, J, b)
      # prob_gt = model.inference()
      #
      # graph['prob_gt'] = prob_gt  # shape N x 2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_val'] = seed_val
      graph['J_std'] = std_J_A
      graph['b_std'] = std_b_A
      graph['name'] = name

      msg_node, msg_adj = get_msg_graph(G)
      msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
      idx_msg_edge = np.transpose(np.nonzero(msg_adj))
      J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

      graph['msg_node'] = msg_node
      graph['idx_msg_edge'] = idx_msg_edge
      graph['J_msg'] = J_msg

      # graph['gibb_burn_in'] = 10000
      # graph['gibb_num_sample'] = 10000
      # graph['block_size'] = 4
      #
      # graph['hmc_burn_in'] = 10000
      # graph['hmc_num_sample'] = 10000
      # graph['trv_time']= 49.5
      #
      # graph['hmc_time'] = hmc_time
      # graph['gibbs_time'] = gibbs_time
      #
      # diff = abs(prob_gibbs - prob_hmc).mean()
      # if diff < 0.02:
      #     file_name = os.path.join(save_dir, 'val', 'pass', 'graph_{}_nn{}_{:07d}.p'.format(name, num_nodes_I, seed))
      # else:
      #     file_name = os.path.join(save_dir, 'val', 'fail', 'graph_{}_nn{}_{:07d}.p'.format(name, num_nodes_I, seed))
      file_name = os.path.join(save_dir, 'val', 'graph_{}_nn{}_{:07d}.p'.format(name, num_nodes_I, seed))
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)
        del graph

# python3 -m dataset.gen_test_II_pc
# Use different seed for train/val/test
if __name__ == '__main__':
    print("Setting environment")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_index', type=str, default=0, help='group_index')
    # parser.add_argument('--index', type=int, default=1, help='index')
    # parser.add_argument('--split_num', type=int, default=1, help='split_num')
    args = parser.parse_args()

    try_num = 0
    cum_count_pass = 0

    p = Pool(processes=20)
    p.starmap(main, zip([i for i in range(20)], [try_num for i in range(20)], [1.5 for i in range(20)]))



