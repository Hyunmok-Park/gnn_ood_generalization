import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
from topology import NetworkTopology, get_msg_graph
from hmc import HMC
from block_gibbs import BlockGibbs
from multiprocessing import Manager, Process, Pool
from scipy.sparse import coo_matrix
import argparse
import math
import time
from tqdm import tqdm
import networkx as nx
from itertools import repeat
import shutil
from itertools import product
from gt_inference import Enumerate

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))


def main(index, try_num):
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
  num_nodes_I = 16
  std_J_A = 0.6
  std_b_A = 0.25
  num_graphs_test = 100

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  block_size = 4

  hmc_burn_in = 10000
  hmc_num_sample = 10000
  trv_time = 49.5

  print('Generating training graphs!')

  save_dir = 'data_temp/exp2_V16_test_0.6/try_{}'.format(try_num)
  J_list = pickle.load(open("data_temp/V16_test_{}.p".format(try_num), "rb"))['J_list']
  name_list_full = pickle.load(open("data_temp/V16_test_{}.p".format(try_num), "rb"))['name_list']

  n = math.ceil(len(J_list)/ 30)

  tt_list = J_list[n * index:n * (index+1)]
  name_list = name_list_full[n * index:n * (index+1)]
  seed_list = [i+2000*try_num for i in range(2000)][n * index:n * (index+1)]

  print("total length", len(J_list), len(tt_list))
  del J_list, name_list_full

  for tt, seed, name in zip(tt_list, seed_list, name_list):
      seed_test = int(str(3333) + str(seed))
      npr = np.random.RandomState(seed=seed_test)

      # print(tt)
      #############################################################################
      # Generate Training Graphs
      #############################################################################
      # here seed only affects random graphs
      print('Generating training graphs!')
      # topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)

      try:
          mkdir(os.path.join(save_dir))
          mkdir(os.path.join(save_dir, 'pass'))
          mkdir(os.path.join(save_dir, 'fail'))
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
      # model = BlockGibbs(G, J, b, block_method='gibbs', block_size=block_size, seed=seed_test)
      # prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
      # gibbs_time = time.time() - tik1
      #
      # # Hamiltonian Monte Carlo
      # tik2 = time.time()
      # model = HMC(W, J, b, seed=seed_test)
      # prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)
      # hmc_time = time.time() - tik2

      # graph['prob_hmc'] = prob_hmc
      # graph['prob_gibbs'] = prob_gibbs

      # Enumerate
      model = Enumerate(W, J, b)
      prob_gt = model.inference()

      graph['prob_gt'] = prob_gt  # shape N x 2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_test'] = seed_test
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
      diff = 0
      # diff = abs(prob_gibbs - prob_hmc).mean()
      if diff < 0.02:
          file_name = os.path.join(save_dir, 'pass', 'graph_{}_nn{}_{:07d}.p'.format(name, num_nodes_I, seed))
      else:
          file_name = os.path.join(save_dir, 'fail', 'graph_{}_nn{}_{:07d}.p'.format(name, num_nodes_I, seed))

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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--group_index', type=str, default=1, help='group_index')
    # parser.add_argument('--index', type=int, default=1, help='index')
    # parser.add_argument('--split_num', type=int, default=1, help='split_num')
    # args = parser.parse_args()

    try_num = 0
    cum_count_pass = 0
    num_nodes_I = 16
    while True:
        p = Pool(processes=30)
        p.starmap(main, zip([i for i in range(30)], [try_num for i in range(30)]))

        J_list = []
        name_list = []
        new_pickle = {}
        for i in os.listdir('data_temp/exp2_test_16_0.6/try_{}/fail'.format(try_num)):
            data_path = os.path.join('data_temp/exp2_test_16_0.6/try_{}/fail'.format(try_num), i)
            data = pickle.load(open(data_path, "rb"))
            J = data['J'].todense()
            for ii,jj in product(range(num_nodes_I), range(num_nodes_I)):
                if J[ii,jj] != 0:
                    J[ii,jj] = 1
            J_list.append(J)
            name_list.append(data['name'])
        new_pickle['J_list'] = J_list
        new_pickle['name_list'] = name_list

        count_pass = len(os.listdir('data_temp/exp2_test_16_0.6/try_{}/pass'.format(try_num)))
        count_fail = len(os.listdir('data_temp/exp2_test_16_0.6/try_{}/fail'.format(try_num)))
        cum_count_pass += count_pass

        try_num += 1
        with open("data_temp/V16_test_{}.p".format(try_num), "wb") as f:
            pickle.dump(new_pickle, f)
            del new_pickle

        with open("data_temp/process.txt" ,"a") as f:
            f.write("======Try {} report======\n".format(try_num - 1))
            f.write("Pass : {}\n".format(count_pass))
            f.write("Fail : {}\n".format(count_fail))
            f.write("Current pass : {}\n".format(cum_count_pass))

        # print("======Try {} report======".format(try_num-1))
        # print("Pass : {}".format(count_pass))
        # print("Fail : {}".format(count_fail))
        # print("Current pass : {}".format(cum_count_pass))

        if count_fail == 0:
            break

