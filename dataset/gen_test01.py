import os
import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.gt_inference import Enumerate
from model.block_gibbs import BlockGibbs
from model.hmc import HMC
from time import time
from scipy.sparse import coo_matrix
import argparse

#===============================================================
import datajoint as dj
schema = dj.schema('kijung-test01')  # Change this

@schema
class Sample(dj.Manual):
  definition = """
    sample_id : int unsigned
    """
@schema
class SampleComputed(dj.Computed):
  definition = """
    -> Sample
    """
  def make(self, key):
    sample_id = key['sample_id']
    # Call main function to start computation
    main(sample_id)
    self.insert1(key)  # Just to mark it as completed
#===============================================================

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main(sample_id):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
  """
  seed_test = int(str(3333) + str(sample_id))
  std_J_A = 0.9
  std_b_A = 0.25
  num_nodes_I = 6**2
  num_graphs_test = 20

  # save_dir = '../data_grid/'
  save_dir = '/mnt/scratch07/kijung/data_hmc/'

  gibb_burn_in = 10000
  gibb_num_sample = 10000
  bs = 4

  hmc_num_sample = 10000
  hmc_burn_in = 10000
  trv_time = 49.5

  topology_list = [
    'path', 'binarytree', 'cycle', 'grid', 'circladder', 'ladder', 'cylinder', 'torus',
    'trigrid', 'trikite', 'trilattice', 'barbell646', 'barbell808', 'bipartite', 'complete'
  ]
  topology_list = [
    'barbell646', 'barbell808', 'bipartite', 'complete'
  ]
  test_name = 'test_01'
  npr = np.random.RandomState(seed=seed_test)

  #############################################################################
  # Generate Training Graphs
  #############################################################################
  # here seed only affects random graphs
  print('Generating test graphs!')
  topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_test)
  try:
    mkdir(os.path.join(save_dir, test_name))
  except OSError:
    pass

  for tt in topology_list:
    print(tt)
    graph = {}
    if tt in ['binarytree', 'bipartite', 'circladder', 'complete', 'cycle', 'ellcube', 'grid', 'ladder', 'path', 'star', 'trigrid', 'trigridv2', 'wheel']:
      G, W = topology.generate(topology=tt)
    elif tt in ['cylinder', 'torus']:
      G, W = topology.generate(topology=tt, argin1=4, argin2=num_nodes_I//4 - 1)
    elif tt in ['diamondcube']:
      G, W = topology.generate(topology=tt, argin1=5, argin2=1)
    elif tt in ['hexlattice']:
      G, W = topology.generate(topology=tt, argin1=2, argin2=2)
    elif tt in ['nonalattice']:
      G, W = topology.generate(topology=tt, argin1=2)
    elif tt in ['trikite']:
      G, W = topology.generate(topology=tt, argin1=num_nodes_I//3)
    elif tt in ['trilattice']:
      G, W = topology.generate(topology=tt, argin1=1, argin2=num_nodes_I-2)
    elif tt in ['barbell808']:
      G, W = topology.generate(topology='barbell', argin1=num_nodes_I//2, argin2=0)
    elif tt in ['barbell646']:
      G, W = topology.generate(topology='barbell', argin1=(num_nodes_I-4)//2, argin2=4)

    J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
    J = (J + J.transpose()) / 2.0
    J = J * W
    b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

    # Enumerate
    # model = Enumerate(W, J, b)
    # prob_gt = model.inference()
    # map_gt = model.inference_map()

    # Block Gibbs
    tic = time()
    model = BlockGibbs(G, J, b, block_method='gibbs', block_size=bs, seed=seed_test)
    prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
    toc = time()
    time_gibbs = toc - tic

    # HMC
    tic = time()
    model = HMC(W, J, b, seed=seed_test)
    prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)
    toc = time()
    time_hmc = toc - tic

    # graph['prob_gt'] = prob_gt # shape N x 2
    # graph['map_gt'] = map_gt  # shape N x 2
    graph['prob_gibbs'] = prob_gibbs  # shape N x 2
    graph['prob_hmc'] = prob_hmc  # shape N x 2
    graph['prob_gt'] = prob_hmc
    graph['J'] = coo_matrix(J)  # shape N X N
    graph['b'] = b  # shape N x 1
    graph['seed_test'] = seed_test
    graph['time_gibbs'] = time_gibbs
    graph['time_hmc'] = time_hmc

    msg_node, msg_adj = get_msg_graph(G)
    msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
    idx_msg_edge = np.transpose(np.nonzero(msg_adj))
    J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

    graph['msg_node'] = msg_node
    graph['idx_msg_edge'] = idx_msg_edge
    graph['J_msg'] = J_msg

    file_name = os.path.join(save_dir, test_name, 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_I, sample_id))
    with open(file_name, 'wb') as f:
      pickle.dump(graph, f)
      del graph


# python3 -m dataset.gen_val
# Use different seed for train/val/test
if __name__ == '__main__':
    SampleComputed().populate(reserve_jobs=True, suppress_errors=False)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    # args = parser.parse_args()
    #
    # main(args.sample_id)
