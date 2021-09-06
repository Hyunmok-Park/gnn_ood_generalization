import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn

__all__ = ['TreeReWeightedMessagePassing']


class TreeReWeightedMessagePassing(nn.Module):

  def __init__(self, config):
    """
      Belief Propagation for Binary Undirected Markov Models

      A: shape N X N, binary adjacency matrix
      J: shape N X N, parameters of pairwise term, must be symmetric
      b: shape N X 1, parameters of unary term
      msg_node: list, nodes of message graph, i.e., edges of original graph
      msg_adj: shape E X E, adjacency matrix of message graph
    """
    super(TreeReWeightedMessagePassing, self).__init__()
    self.max_iter = config.model.max_iter
    self.damping = config.model.damping
    self.num_trees = config.model.num_trees
    self.loss_func = nn.KLDivLoss(reduction='sum')

  # def forward(self, A, J, b, msg_node, msg_adj, target=None, mask=None):
  def forward(self, J, b, msg_node, msg_adj, target=None, mask=None):
    """

      A: shape N X N
      J: shape N X N
      b: shape N X 1
      msg_node: shape T X E X 2
      msg_adj: shape T X E X E
    """
    num_states = 2
    # num_nodes = A.shape[0]
    num_nodes = J.shape[0]
    log_prob = torch.zeros(num_nodes, num_states)  # shape N X 2
    # if A.is_cuda:
    #   log_prob = log_prob.cuda()

    prob_step = []
    for tt in range(self.num_trees):
      num_edges = msg_node[tt].shape[0]
      print("sssssss", msg_node.shape)
      edge_in = msg_node[tt, :, 0]  # shape E X 1
      edge_out = msg_node[tt, :, 1]  # shape E X 1
      edge_np = list(zip(edge_in.data.cpu().numpy(), edge_out.data.cpu().numpy()))

      agg_mask = torch.zeros(num_nodes, num_edges)  # shape N X E
      for ii in range(num_edges):
        agg_mask[edge_out[ii], ii] = 1.0

      # if A.is_cuda:
      #   agg_mask = agg_mask.cuda()

      # unary term, shape N X 2, [+1, -1]
      log_phi = torch.cat([b, -b], dim=1)

      # pairwise term, shape E X 2 X 2, [+1 x +1, +1 x -1, -1 x +1, -1 x -1]
      state = torch.Tensor([[1.0, -1.0, -1.0, 1.0]])
      # if A.is_cuda:
      #   state = state.cuda()
      Je = torch.stack([J[nn] for nn in edge_np]).view(-1, 1)  # shape E X 1
      log_psi = Je.mm(state).view(-1, num_states, num_states)

      # observe mask, shape E X 2 X 2
      if mask is None:
        mask = torch.ones(num_edges, num_states, num_states)
        # if A.is_cuda:
        #   mask = mask.cuda()

      # log message, shape E X 2
      log_msg = torch.log(torch.ones(num_edges, num_states) / float(num_states))
      # if A.is_cuda:
      #   log_msg = log_msg.cuda()

      for ii in range(self.max_iter):
        log_msg = (1 - self.damping) * torch.logsumexp(
            mask * (log_phi[edge_in].unsqueeze(dim=2) + log_psi + torch.mm(
                msg_adj[tt].t(), log_msg.long()).unsqueeze(dim=2).float()),
            dim=1) + self.damping * log_msg
        log_msg = log_msg - torch.logsumexp(log_msg, dim=1, keepdim=True)

      # shape N X 2
      log_prob_tmp = log_phi + agg_mask.mm(log_msg)
      log_prob_tmp = torch.log_softmax(log_prob_tmp, dim=1)
      log_prob = log_prob + log_prob_tmp

      prob_step_ = log_prob / float(self.num_trees)
      prob_step += [np.expand_dims(torch.exp(prob_step_[:, 0]).data.cpu().numpy(), axis=0)]

    prob_step = np.concatenate(prob_step, axis=0).transpose()
    log_prob = log_prob / float(self.num_trees)

    if target is not None:
      loss = self.loss_func(log_prob, target)
    else:
      loss = None

    return torch.exp(log_prob), loss, prob_step
