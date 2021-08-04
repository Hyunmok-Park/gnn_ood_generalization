import numpy as np
import torch
import torch.nn as nn

__all__ = ['MeanField']
EPS = float(np.finfo(np.float32).eps)


class MeanField(nn.Module):

  def __init__(self, config):
    """
      Mean field for Binary Undirected Markov Models

      A: shape N X N, binary adjacency matrix
      J: shape N X N, parameters of pairwise term, must be symmetric
      b: shape N X 1, parameters of unary term
      msg_node: list, nodes of message graph, i.e., edges of original graph
      msg_adj: shape E X E, adjacency matrix of message graph
    """
    super(MeanField, self).__init__()
    self.max_iter = config.model.max_iter
    self.damping = config.model.damping
    self.loss_func = nn.KLDivLoss(reduction='sum')

  # def forward(self, A, J, b, msg_node, msg_adj, target=None, mask=None):
  def forward(self, J, b, msg_node, msg_adj, target=None, mask=None):
    """

      A: shape N X N
      J: shape N X N
      b: shape N X 1
      msg_node: shape E X 2
      msg_adj: shape E X E
    """
    num_states = 2
    # num_nodes = A.shape[0]
    num_nodes = J.shape[0]

    # log message, shape E X 2
    prob = torch.ones(num_nodes, num_states) / 2.0
    # if A.is_cuda:
    #   prob = prob.cuda()

    prob_step = []
    log_prob = torch.log(prob)  # N X 1
    for ii in range(self.max_iter):
      prob = torch.exp(log_prob)
      diag_J = torch.diag(J).view(-1, 1) + EPS      
      tmp_prob = prob[:, 0].view(-1, 1)
      tmp_prob = torch.log(1 - tmp_prob + EPS) + 8 * J.mm(tmp_prob - 1) - 4 * diag_J * (tmp_prob - 1) + 2 * b - 1
      log_prob = torch.cat([tmp_prob, log_prob[:, 1].view(-1, 1)], dim=1)
      log_prob = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)

      prob_step += [np.expand_dims(torch.exp(log_prob[:, 0]).data.cpu().numpy(), axis=0)]

    prob_step = np.concatenate(prob_step, axis=0).transpose()
          
    if target is not None:
      loss = self.loss_func(log_prob, target)
    else:
      loss = None

    return torch.exp(log_prob), loss, prob_step
