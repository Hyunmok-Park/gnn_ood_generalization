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


def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))


def main(sample_id, begin, end):
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
  save_dir = '../data_temp/'

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  block_size = 4

  hmc_burn_in = 10000
  hmc_num_sample = 10000
  trv_time = 49.5

  # topology_list = [
  #     'star', 'binarytree', 'path', 'cycle', 'wheel', 'ladder',
  #     'circladder', 'grid', 'barbell', 'lollipop', 'bipartite',
  #     'tripartite', 'complete'
  # ]

  topology_list = [
    'grid'
  ]
  file_list = [
    'grid_900_0.9'
  ]

  #npr = np.random.RandomState(seed=seed_test)

  #############################################################################
  # Generate Training Graphs
  #############################################################################
  # here seed only affects random graphs
  print('Generating training graphs!')
  #topology = NetworkTopology(num_nodes=num_nodes_II, seed=seed_test)
  # try:
  #   mkdir(os.path.join(save_dir, 'test_II'))
  # except OSError:
  #   pass
  for i in file_list:
      try:
        mkdir(os.path.join(save_dir, i))  #!!
      except OSError:
        pass

  for tt, jj in zip(topology_list, file_list):
    print(tt)
    for ii in tqdm(range(begin, end)):
      sample_id = ii  #
      seed_test = int(str(3333) + str(sample_id))  #
      topology = NetworkTopology(num_nodes=num_nodes_II, seed=seed_test)  #
      npr = np.random.RandomState(seed=seed_test)  #
      graph = {}
      G, W = topology.generate(topology=tt)
      J = npr.normal(0, std_J_A, size=[num_nodes_II, num_nodes_II])
      J = (J + J.transpose()) / 2.0
      J = J * W
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

      # seed_test2 = int(str(3334) + str(sample_id) + str(sample_id))
      # model2 = HMC(W, J, b, seed=seed_test2)
      # prob_hmc2 = model2.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)


      # if (J >= 0).all():
      #     model = MinCut(G, J, b)
      #     map_gt = model.inference()
      #     graph['map_gt'] = map_gt  # shape N x 2

      graph['prob_gibbs'] = prob_gibbs  # shape N x 2
      graph['prob_hmc'] = prob_hmc
      # graph['prob_hmc2'] = prob_hmc2
      # graph['prob_log'] = prob_log
      # graph['prob_log2'] = prob_log2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_test'] = seed_test
      graph['J_std'] = std_J_A

      msg_node, msg_adj = get_msg_graph(G)
      msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
      idx_msg_edge = np.transpose(np.nonzero(msg_adj))
      J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

      graph['msg_node'] = msg_node
      # graph['msg_adj'] = msg_adj
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

      #file_name = os.path.join(save_dir, 'test_II', 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_II, sample_id))
      file_name = os.path.join(save_dir, jj, 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_II, sample_id))
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)
        del graph


# python3 -m dataset.gen_test_II
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    args = parser.parse_args()

    pr1 = Process(target=main, args=(1, 0, 2))
    pr2 = Process(target=main, args=(2, 2, 4))
    pr3 = Process(target=main, args=(3, 4, 6))
    pr4 = Process(target=main, args=(4, 6, 8))
    pr5 = Process(target=main, args=(5, 8, 10))
    pr6 = Process(target=main, args=(6, 10, 12))
    pr7 = Process(target=main, args=(7, 12, 14))
    pr8 = Process(target=main, args=(8, 14, 16))
    pr9 = Process(target=main, args=(9, 16, 18))
    pr10 = Process(target=main, args=(10, 18, 20))
    pr11 = Process(target=main, args=(11, 20, 22))
    pr12 = Process(target=main, args=(12, 22, 24))
    pr13 = Process(target=main, args=(13, 24, 26))
    pr14 = Process(target=main, args=(14, 26, 28))
    pr15 = Process(target=main, args=(15, 28, 30))
    pr16 = Process(target=main, args=(16, 30, 32))

    pr1.start()
    pr2.start()
    pr3.start()
    pr4.start()
    pr5.start()
    pr6.start()
    pr7.start()
    pr8.start()
    pr9.start()
    pr10.start()
    pr11.start()
    pr12.start()
    pr13.start()
    pr14.start()
    pr15.start()
    pr16.start()

    pr1.join()
    pr2.join()
    pr3.join()
    pr4.join()
    pr5.join()
    pr6.join()
    pr7.join()
    pr8.join()
    pr9.join()
    pr10.join()
    pr11.join()
    pr12.join()
    pr13.join()
    pr14.join()
    pr15.join()
    pr16.join()
    # main(args.sample_id)
