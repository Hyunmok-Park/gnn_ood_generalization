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

  std_J_A = 0.6
  std_b_A = 0.25
  num_nodes_I = 16
  num_graphs_train = 100
  topology_list = [
    'binarytree', 'grid', 'trigrid', 'bipartite'
  ]

  print('Generating training graphs!')

  for tt in topology_list:
    print(tt)
    save_dir = '/home/ubuntu/pycharm_project_ACD/codebase/data_temp_BP/nn_16/0.6_0.6/{}'.format(tt)
    try:
      mkdir(os.path.join(save_dir, 'val'))
    except OSError:
      pass

    for ii in range(begin, end):
      sample_id = ii#
      seed_val = int(str(2222) + str(sample_id))#
      topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_val)#
      npr = np.random.RandomState(seed=seed_val)#
      graph = {}
      G, W = topology.generate(topology=tt)
      J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
      J = (J + J.transpose()) / 2.0
      J = J * W
      b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

      # Enumerate
      model = Enumerate(W, J, b)
      prob_gt = model.inference()
      # map_gt = model.inference_map()

      graph['prob_gt'] = prob_gt # shape N x 2
      # graph['map_gt'] = map_gt  # shape N x 2
      graph['J'] = coo_matrix(J)  # shape N X N
      graph['b'] = b  # shape N x 1
      graph['seed_val'] = seed_val
      graph['stdJ'] = std_J_A

      msg_node, msg_adj = get_msg_graph(G)
      msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
      idx_msg_edge = np.transpose(np.nonzero(msg_adj))
      J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

      graph['msg_node'] = msg_node
      graph['idx_msg_edge'] = idx_msg_edge
      graph['J_msg'] = J_msg

      # BeliefPropagation
      bp_prob, bp_step = BeliefPropagation(J, b, msg_node, msg_adj).inference(max_iter = 10, damping = 0.0, observe_mask=None)
      graph['bp_step'] = bp_step

      file_name = os.path.join(save_dir, 'val', 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_I, sample_id))
      with open(file_name, 'wb') as f:
        pickle.dump(graph, f)
        del graph


# python3 -m dataset.gen_train
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    args = parser.parse_args()

    jobs = []
    for i in range(10):
        begin, end = 10 * i, 10 * (i + 1)
        p = Process(target=main, args=(begin, end))
        jobs.append(p)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()
