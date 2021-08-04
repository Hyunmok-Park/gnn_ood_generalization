import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.hmc import HMC
from model.block_gibbs import BlockGibbs
from multiprocessing import Manager, Process, Pool
from scipy.sparse import coo_matrix
import argparse
import math
import time
import os
from tqdm import tqdm
import networkx as nx
from itertools import repeat
import shutil
from itertools import product
from model.gt_inference import Enumerate

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main(tt, stdJ, seed):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
  """
  #seed_train = int(str(1111)+str(sample_id))
  num_nodes_I = 100
  std_J_A = stdJ
  std_b_A = 0.25
  num_graphs_train = 9

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  block_size = 4

  hmc_burn_in = 10000
  hmc_num_sample = 10000
  trv_time = 49.5

  save_dir = '/home/ubuntu/TorchGNN_project/data_temp/exp1_test/test_{}/{}'.format(std_J_A, num_nodes_I)

  #npr = np.random.RandomState(seed=seed_train)
  #############################################################################
  # Generate Training Graphs
  #############################################################################
  # here seed only affects random graphs
  print('Generating training graphs!')
  #topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)

  try:
      mkdir(os.path.join(save_dir, "{}_{}".format(tt, std_J_A)))
  except OSError:
      pass

  sample_id = seed
  seed_test = int(str(3333) + str(sample_id))#
  topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_test)#
  npr = np.random.RandomState(seed=seed_test)#
  graph = {}
  G, W = topology.generate(topology=tt)
  # J = npr.uniform(high = std_J_A/ np.sqrt(num_nodes_I), size = [num_nodes_I, num_nodes_I])
  # J = npr.normal(0, std_J_A / np.sqrt(num_nodes_I), size=[num_nodes_I, num_nodes_I])
  J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
  J = (J + J.transpose()) / 2.0
  J = J * W
  #b = npr.uniform(low = -2, high = 2, size=[num_nodes_I, 1])
  b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

  # Enumerate
  # model = Enumerate(W, J, b)
  # prob_gt = model.inference()
  # map_gt = model.inference_map()

  # Block Gibbs Sampling
  # tik1 = time.time()
  # model = BlockGibbs(G, J, b, block_method='gibbs', block_size=block_size, seed=seed_test)
  # prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
  # gibbs_time = time.time() - tik1

  # Hamiltonian Monte Carlo
  tik2 = time.time()
  model = HMC(W, J, b, seed=seed_test)
  prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)
  hmc_time = time.time() - tik2

  graph['prob_hmc'] = prob_hmc
  # graph['prob_gibbs'] = prob_gibbs

  # graph['prob_gt'] = prob_gt # shape N x 2
  # graph['map_gt'] = map_gt  # shape N x 2
  graph['J'] = coo_matrix(J)  # shape N X N
  graph['b'] = b  # shape N x 1
  graph['seed_test'] = seed_test
  graph['stdJ'] = std_J_A
  graph['stdb'] = std_b_A

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

  # diff = abs(prob_gibbs - prob_hmc).mean()
  diff = 0.01
  if diff < 0.02:
      file_name = os.path.join(save_dir, "{}_{}".format(tt, std_J_A), 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_I, sample_id))
      with open(file_name, 'wb') as f:
          pickle.dump(graph, f)
  else:
      pass

# python3 -m dataset.gen_train
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    args = parser.parse_args()

    for iii, tt in enumerate(['star', 'lollipop', 'wheel']):
        for stdJ in [0.3,0.6,0.9]:
            try_num = 0
            while True:
                p = Pool(processes=16)
                p.starmap(main, zip([tt for i in range(16)], [stdJ for i in range(16)], [i for i in range(100+iii*1000+try_num*16,116+iii*1000+try_num*16)]))

                path = "/home/ubuntu/TorchGNN_project/data_temp/exp1_test/test_{}/100/{}_{}".format(stdJ,tt,stdJ)
                if len(os.listdir(path)) >= 10:
                    break
                else:
                    try_num += 1


