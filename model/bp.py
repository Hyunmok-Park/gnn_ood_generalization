import numpy as np
from scipy.special import logsumexp, kl_div
import torch
import torch.nn as nn

EPS = float(np.finfo(np.float32).eps)
__all__ = ['BeliefPropagation', 'BeliefPropagationTorch']

class BeliefPropagation(object):

  def __init__(self, J, b, msg_node, msg_adj):
    """
      Belief Propagation for Binary Undirected Markov Models

      A: shape N X N, binary adjacency matrix
      J: shape N X N, parameters of pairwise term, must be symmetric
      b: shape N X 1, parameters of unary term
      msg_node: list, nodes of message graph, i.e., edges of original graph
      msg_adj: shape E X E, adjacency matrix of message graph
    """
    super(BeliefPropagation, self).__init__()
    self.num_nodes = J.shape[0]
    self.num_edges = len(msg_node)
    self.num_states = 2
    self.msg_adj = msg_adj # adjacency matrix of message graph, shape E X E For step 1
    self.edge_in = np.array([nn[0] for nn in msg_node])  # shape E X 1
    self.edge_out = np.array([nn[1] for nn in msg_node])  # shape E X 1
    # self.edge_in = msg_node[:, 0].numpy()  # shape E X 1
    # self.edge_out = msg_node[:, 1].numpy()  # shape E X 1
    self.edge_np = list(zip(self.edge_in, self.edge_out))

    self.agg_mask = np.zeros([self.num_nodes, self.num_edges])  # shape N X E : incoming message edge indices
    for ii in range(self.num_edges):
      self.agg_mask[self.edge_out[ii]][ii] = 1.0 # For step 3

    # 1. unary term phi = exp(bx), shape N X 2, [+1, -1]
    self.log_phi = np.concatenate([b, -b], axis=1)

    # 2. pairwise term psi = exp(Jxixj), shape E X 2 X 2, [+1 x +1, +1 x -1, -1 x +1, -1 x -1]
    state = np.array([[1.0], [-1.0]])
    state = state.dot(state.transpose())
    Je = np.array([J[nn] for nn in self.edge_np]).reshape(-1, 1)  # shape E X 1
    self.log_psi = Je.dot(state.reshape(1, -1)).reshape(-1, self.num_states, self.num_states) # shape E X 2 X 2

  def inference(self, max_iter = 10, kl_order = 'pq', damping = 0.0, target=None, observe_mask=None):
    # Useless: observe_mask, shape E X 2 X 2
    if observe_mask is None:
      observe_mask = np.ones((self.num_edges, self.num_states, self.num_states))

    # log message, shape E X 2
    log_msg = np.log(np.ones((self.num_edges, self.num_states)))
    prob_step = []
    for ii in range(max_iter):
      # 1. Message Update : log(psi) + log(phi) + sum(log messages) : (E X 2 X 2) + (E X 1 X 2) + (E X 1 X 2)
      log_temp = self.log_psi + np.expand_dims(self.log_phi[self.edge_in], axis=1) + np.expand_dims(np.dot(self.msg_adj.transpose(), log_msg), axis=1)
      log_msg = (1 - damping) * logsumexp(observe_mask * log_temp, axis=2) + damping * log_msg
      # 2. Message Normalization
      log_msg = log_msg - logsumexp(log_msg, axis=1, keepdims=True)


      # 3. Belief Update: shape N X 2
      log_prob = self.log_phi + self.agg_mask.dot(log_msg)
      # 4. Belief Normalization
      log_prob = log_prob - logsumexp(log_prob, axis=1, keepdims=True)
      prob = np.exp(log_prob)
      prob_step.append(prob[:,0].tolist())

    # if kl_order == 'pq' and target is not None:
    #   loss = kl_div(target.double().numpy(), prob).sum()
    # elif kl_order == 'qp' and target is not None:
    #   loss = kl_div(prob, target.double().numpy()).sum()
    # else:
    #   loss = None
    prob_step = np.array(prob_step).transpose()
    return prob, prob_step #, loss


class BeliefPropagationTorch(nn.Module):

  def __init__(self, config):
    """
      Belief Propagation for Binary Undirected Markov Models

      A: shape N X N, binary adjacency matrix
      J: shape N X N, parameters of pairwise term, must be symmetric
      b: shape N X 1, parameters of unary term
      msg_node: list, nodes of message graph, i.e., edges of original graph
      msg_adj: shape E X E, adjacency matrix of message graph
    """
    super(BeliefPropagationTorch, self).__init__()
    self.max_iter = config.model.max_iter
    self.damping = config.model.damping
    self.loss_func = nn.KLDivLoss(reduction='sum')
    self.kl_order = config.model.kl_order

  def forward(self, J, b, msg_node, msg_adj, target=None, mask=None):
    """
      A: shape N X N
      J: shape N X N
      b: shape N X 1
      msg_node: shape E X 2
      msg_adj: shape E X E
    """
    num_states = 2
    num_nodes = J.shape[0]
    num_edges = msg_node.shape[0]
    edge_in = msg_node[:, 0]  # shape E X 1
    edge_out = msg_node[:, 1]  # shape E X 1
    edge_np = list(zip(edge_in.data.cpu().numpy(), edge_out.data.cpu().numpy()))

    agg_mask = torch.zeros(num_nodes, num_edges)  # shape N X E : incoming message edge indices
    for ii in range(num_edges):
      agg_mask[edge_out[ii], ii] = 1.0 # For step 3
    if J.is_cuda:
      agg_mask = agg_mask.cuda()

    # 1. unary term phi = exp(bx), shape N X 2, [+1, -1]
    log_phi = torch.cat([b, -b], dim=1)

    # 2. pairwise term psi = exp(Jxixj), shape E X 2 X 2, [+1 x +1, +1 x -1, -1 x +1, -1 x -1]
    state = torch.Tensor([[1.0], [-1.0]])
    state = state.mm(state.t())
    if J.is_cuda:
      state = state.cuda()
    Je = torch.stack([J[nn] for nn in edge_np]).view(-1, 1)  # shape E X 1
    log_psi = Je.mm(state.view(1,-1)).view(-1, num_states, num_states)

    # observe mask, shape E X 2 X 2
    if mask is None:
      mask = torch.ones(num_edges, num_states, num_states)
      if J.is_cuda:
        mask = mask.cuda()

    # log message, shape E X 2
    log_msg = torch.log(torch.ones(num_edges, num_states))
    if J.is_cuda:
      log_msg = log_msg.cuda()

    prob_step = []
    for ii in range(self.max_iter):
      # 1. Message Update : log(psi) + log(phi) + sum(log messages) : (E X 2 X 2) + (E X 1 X 2) + (E X 1 X 2)
      log_temp = log_psi + log_phi[edge_in].unsqueeze(dim=1) + torch.mm(msg_adj.t(), log_msg.long()).unsqueeze(dim=1).float()
      log_msg = (1 - self.damping) * torch.logsumexp(mask * log_temp, dim=2) + self.damping * log_msg
      # 2. Message Normalization
      log_msg = log_msg - torch.logsumexp(log_msg, dim=1, keepdim=True)

      # 3. Belief Update: shape N X 2
      log_prob = log_phi + agg_mask.mm(log_msg)
      # 4. Belief Normalization
      log_prob = torch.log_softmax(log_prob, dim=1)

      prob_step += [np.expand_dims(torch.exp(log_prob[:, 0]).data.cpu().numpy(), axis=0)]

    prob_step = np.concatenate(prob_step, axis=0).transpose()

    if self.kl_order == 'pq' and target is not None:
      loss = self.loss_func(log_prob, target)
    elif self.kl_order == 'qp' and target is not None:
      loss = self.loss_func(torch.log(target + EPS), torch.exp(log_prob))
    else:
      loss = None

    return torch.exp(log_prob), loss, prob_step


# python3 -m model.bp
if __name__ == '__main__':
  import networkx as nx
  from utils.topology import get_msg_graph
  from model.gt_inference import Enumerate
  npr = np.random.RandomState(seed=1234)

  A = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
  J = npr.randn(5, 5)
  J = (J + J.transpose()) * 0.5
  b = npr.randn(5, 1) * 0.5

  msg_node, msg_adj = get_msg_graph(nx.from_numpy_matrix(A))
  model = BeliefPropagation(J, b, msg_node, msg_adj)
  P = model.inference()
  model = Enumerate(A, J, b)
  P_gt = model.inference()

  print('diff = {}'.format(np.linalg.norm(P - P_gt)))
  print(P, P_gt)