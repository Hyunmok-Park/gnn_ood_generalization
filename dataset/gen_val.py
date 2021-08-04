import os
import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.gt_inference import Enumerate
from scipy.sparse import coo_matrix
import argparse
from multiprocessing import Process
from model.bp import BeliefPropagation

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main(begin , end):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
  """
  #seed_train = int(str(1111)+str(sample_id))
  num_nodes_I = 9
  std_J_A = 1/3
  std_b_A = 0.25
  num_graphs_train = 9
  save_dir = '/home/ubuntu/pycharm_project_torchgnn/data_temp/V=9/all_J0.33_b0.25_seed'

  topology_list = [
    'path', 'binarytree', 'cycle', 'grid', 'circladder', 'ladder', 'barbell808', 'bipartite', 'complete', 'lollipop', 'star', 'tripartite', 'wheel'
  ]
  # topology_list = [
  #     'path', 'binarytree', 'cycle', 'grid', 'circladder', 'ladder', 'barbell808', 'bipartite', 'complete', 'torus', 'barbell646',
  #     'trikite','trilattice', 'cylinder', 'trigrid'
  # ]

  #npr = np.random.RandomState(seed=seed_train)

  #############################################################################
  # Generate Training Graphs
  #############################################################################
  # here seed only affects random graphs
  print('Generating training graphs!')
  #topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)

  for tt_idx, tt in enumerate(topology_list):
    print(tt)
    try:
        mkdir(os.path.join(save_dir, "val"))
    except OSError:
        pass

    for ii in range(begin, end):
      sample_id = ii + 10*tt_idx#
      seed_train = int(str(2222) + str(sample_id))#
      topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)#
      npr = np.random.RandomState(seed=seed_train)#
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
      model = Enumerate(W, J, b)
      prob_gt = model.inference()
      # map_gt = model.inference_map()

      graph['prob_gt'] = prob_gt # shape N x 2
      # graph['map_gt'] = map_gt  # shape N x 2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_val'] = seed_train
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

      file_name = os.path.join(save_dir, "val", 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_I, sample_id))
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)

# python3 -m dataset.gen_train
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    args = parser.parse_args()

    jobs = []
    for i in range(10):
        begin, end = 1 * i, 1 * (i + 1)

        p = Process(target=main, args=(begin, end))
        jobs.append(p)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()
