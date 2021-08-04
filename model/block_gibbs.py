import networkx as nx
import numpy as np
from tqdm import tqdm
from model.bp import BeliefPropagation as BP
from utils.topology import get_msg_graph

from scipy.special import logsumexp

EPS = float(np.finfo(np.float32).eps)


class BlockGibbs(object):

  def __init__(self, G, J, b, seed=1234, block_size=2, block_method='MST'):
    """
      Blocked Gibbs Sampling for Binary Markov Models

      G: networkx graph, graph
      J: shape N X N, parameters of pairwise term
      b: shape N X 1, parameters of unary term
    """
    super(BlockGibbs, self).__init__()
    self.G = G
    self.A = nx.adjacency_matrix(G).todense()
    self.J = J
    self.b = b
    self.num_nodes = self.A.shape[0]
    self.num_states = 2
    self.states = [1.0, -1.0]
    self.seed = seed
    self.block_size = block_size
    self.block_method = block_method
    self.npr = np.random.RandomState(seed=seed)
    assert self.block_size <= self.num_nodes

  def sample_block_Ising(self, x, block_idx):
    num_nodes = len(block_idx)
    node_states = [np.array(self.states) for _ in range(num_nodes)]
    grid_list = np.meshgrid(*node_states)
    grid_list = [nn.flatten() for nn in grid_list]
    prob = np.zeros((num_nodes, self.num_states))
    sample_x = np.zeros((num_nodes, 1))
    J = np.multiply(self.J, self.A)

    for ii in range(num_nodes):
      log_prob = [[] for _ in range(len(self.states))]
      for jj, ss in enumerate(self.states):
        idx = np.where(grid_list[ii] == ss)[0]

        for kk in range(len(idx)):
          x[block_idx] = np.array([ll[idx[kk]] for ll in grid_list]).reshape(-1, 1)  # shape N X 1
          log_prob[jj] += [x.transpose().dot(self.b) + 0.5 * x.transpose().dot(J).dot(x)]

        prob[ii][jj] = logsumexp(np.concatenate(log_prob[jj], axis=1), axis=1) # still log marginal prob for each state

    prob = np.exp(prob - logsumexp(prob, axis=1).reshape(-1,1))

    for ii in range(num_nodes):
      sample_x[ii] = self.npr.choice([1.0, -1.0], size=1, p=prob[ii])

    x[block_idx] = sample_x
    return x

  def inference(self, burn_in=5000, num_sample=100, sample_gap=50):

    if self.block_method == 'MST':
      prob_avg = np.zeros((self.num_nodes, self.num_states))
      for _ in range(num_sample):
        # generate MST
        W = self.npr.rand(self.num_nodes, self.num_nodes)
        W = np.multiply(W, self.A)
        G = nx.from_numpy_matrix(W).to_undirected()
        MST = nx.minimum_spanning_tree(G)

        # run BP
        msg_node, msg_adj = get_msg_graph(MST)
        A = nx.adjacency_matrix(MST).todense()
        A[A > 0] = 1.0
        J = np.multiply(self.J, A)
        model = BP(J, self.b, msg_node, msg_adj)
        prob = model.inference(max_iter = 10)
        prob_avg += prob

      return prob_avg / float(num_sample)
    else:
      state = self.npr.choice([1.0, -1.0], size=self.num_nodes).reshape(-1, 1)
      # burn in
      for _ in range(burn_in):
        # generate block
        block_idx = self.npr.choice(self.num_nodes, size=self.block_size, replace=False)
        # sample
        state = self.sample_block_Ising(state, block_idx)

      # sampling
      final_state = []
      print("GIBBS")
      for ii in tqdm(range(0, num_sample * sample_gap)):
        # generate block
        block_idx = self.npr.choice(self.num_nodes, size=self.block_size, replace=False)
        # sample
        state = self.sample_block_Ising(state, block_idx)

        if ii % sample_gap == 0:
          final_state += [state.copy()]

      final_state = np.concatenate(final_state, axis=1)
      prob = (final_state == 1).sum(axis=1) / float(final_state.shape[1])

      return np.stack([prob, 1 - prob], axis=0).transpose()


# c
if __name__ == '__main__':
  from model.gt_inference import Enumerate

  npr = np.random.RandomState(seed=1234)
  num_node = 9
  A = np.ones([num_node,num_node]) - np.eye(num_node)
  J = npr.normal(0, 1.0, size=[num_node,num_node])
  J = (J + J.transpose()) * 0.5
  b = npr.normal(0, 0.25, size=[num_node,1])
  G = nx.from_numpy_matrix(A)

  comb = num_node * (num_node-1) // 2
  model = BlockGibbs(G, J, b, block_method='gibbs', block_size=9)
  P = model.inference(burn_in = 10000, num_sample = 100, sample_gap = 50)
  model = Enumerate(A, J, b)
  P_gt = model.inference()
  print('diff_mu = {}'.format(np.linalg.norm(P - P_gt, axis=1).mean()))
  print('diff_sd = {}'.format(np.linalg.norm(P - P_gt, axis=1).std()))
