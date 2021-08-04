import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import block_diag
import torch
import torch.nn as nn
from numba import jit
from mask_rcnn.operators.modules.unsorted_segment_sum import UnsortedSegmentSumFunction
unsorted_segment_sum = UnsortedSegmentSumFunction.apply

__all__ = ['CRF']
EPS = float(np.finfo(np.float32).eps)


class ConditionalRandomField(nn.Module):

  def __init__(self, max_iter=10, damping=0.9):
    """
            Conditional Random Fields
            msg_node: list, nodes of message graph, i.e., edges of original graph
            msg_adj: shape E X E, adjacency matrix of message graph
        """
    super(ConditionalRandomField, self).__init__()
    self.max_iter = max_iter
    self.damping = damping
    ### mean-field
    self.alpha = nn.Parameter(torch.Tensor([[80, 80]]).float())
    self.beta = nn.Parameter(torch.Tensor([[5, 5, 5]]).float())
    self.gamma = nn.Parameter(torch.Tensor([[3, 3]]).float())
    self.w1 = nn.Parameter(torch.ones(1).float() * 1.0e-2)
    self.w2 = nn.Parameter(torch.ones(1).float() * 1.0e-2)

    ### max-product
    # self.alpha = nn.Parameter(torch.Tensor([[80, 80]]).float())
    # self.beta = nn.Parameter(torch.Tensor([[5, 5, 5]]).float())
    # self.gamma = nn.Parameter(torch.Tensor([[3, 3]]).float())
    # self.w1 = nn.Parameter(torch.ones(1).float() * 1.0e-2)
    # self.w2 = nn.Parameter(torch.ones(1).float() * 1.0e-2)

    self.register_buffer('compatible_mask',
                         torch.ones(500, 500) - torch.eye(500))

  # def forward(self, feat, img, incidence_mat, dist_diff, msg_node, msg_adj):
  #     return self.max_product(feat, img, incidence_mat, dist_diff, msg_node, msg_adj)

  def forward(self, feat, img, dist_diff, msg_node):
    return self.mean_field(feat, img, dist_diff, msg_node)

  def max_product(self, feat, img, incidence_mat, dist_diff, msg_node, msg_adj):
    """
            Number of nodes: N = B X H X W

            Args:
                feat: shape B X K X H X W
                img: shape B X 3 X H X W
                incidence_mat: sparse float tensor, shape N X E
                dist_diff: float tensor, shape E X 2
                intensity_diff: float tensor, shape E X 3
                msg_node: long tensor, shape E X 2
                msg_adj: sparse float tensor, shape E X E, transposed

            Return:
                log_prob: shape B X K X H X W
        """
    if self.training:
      # Use GPU
      msg_node = torch.from_numpy(msg_node)
      num_states = feat.shape[1]
      num_nodes = feat.shape[0] * feat.shape[2] * feat.shape[3]
      num_edges = msg_node.shape[0]
      edge_in = msg_node[:, 0]  # shape E X 1

      # pairwise term, shape E X K X K
      img = img.permute(0, 2, 3, 1).reshape(-1, 3)
      intensity_diff = (
          img[msg_node[:, 0], :] - img[msg_node[:, 1], :])**2  # shape E X 3
      compatible_mask = self.compatible_mask[:num_states, :num_states]
      compatible_mask[-1, :] = compatible_mask[-1, :] * 10.0
      compatible_mask[:, -1] = compatible_mask[:, -1] * 10.0
      log_psi = self.w1.abs() * torch.exp(
          -(dist_diff / (self.alpha**2)).sum(dim=1) -
          (intensity_diff / (self.beta**2)).sum(dim=1)) + self.w2.abs(
          ) * torch.exp(-(dist_diff / (self.gamma**2)).sum(dim=1))
      log_psi = -log_psi.view(-1, 1, 1) * compatible_mask.view(
          1, num_states, num_states)

      # log message, shape E X K
      log_msg = torch.log(
          torch.ones(num_edges, num_states) / float(num_states)).to(feat.device)

      # unary term, shape N X K
      log_phi = torch.softmax(
          feat.permute(0, 2, 3, 1).contiguous().view(num_nodes, -1),
          dim=1).log()

      intermediate_result = log_phi[edge_in].unsqueeze(dim=2) + log_psi
      # import pdb; pdb.set_trace()
      for ii in range(self.max_iter):
        log_msg = (1 - self.damping) * torch.max(
            intermediate_result + torch.mm(msg_adj, log_msg).unsqueeze(dim=2),
            dim=1)[0] + self.damping * log_msg
        log_msg = log_msg - torch.logsumexp(log_msg, dim=1, keepdim=True)

      # shape B X K X H X W
      log_prob = log_phi + torch.mm(incidence_mat, log_msg)
      log_prob = torch.log_softmax(log_prob, dim=1)
      log_prob = log_prob.view(feat.shape[0], feat.shape[2], feat.shape[3],
                               -1).permute(0, 3, 1, 2).contiguous()

      return log_prob
    else:
      # use CPU
      msg_node = torch.from_numpy(msg_node)
      num_states = feat.shape[1]
      num_nodes = feat.shape[0] * feat.shape[2] * feat.shape[3]
      num_edges = msg_node.shape[0]
      edge_in = msg_node[:, 0]  # shape E X 1

      # pairwise term, shape E X K X K
      img = img.permute(0, 2, 3, 1).reshape(-1, 3).data.cpu()
      intensity_diff = (
          img[msg_node[:, 0], :] - img[msg_node[:, 1], :])**2  # shape E X 3
      compatible_mask = self.compatible_mask[:num_states, :
                                             num_states].data.cpu()
      compatible_mask[-1, :] = compatible_mask[-1, :] * 1.0e+2
      compatible_mask[:, -1] = compatible_mask[:, -1] * 1.0e+2
      log_psi = self.w1.abs().data.cpu() * torch.exp(
          -(dist_diff.data.cpu() / (self.alpha.data.cpu()**2)).sum(dim=1) -
          (intensity_diff /
           (self.beta.data.cpu()**2)).sum(dim=1)) + self.w2.abs().data.cpu(
           ) * torch.exp(-(dist_diff.data.cpu() /
                           (self.gamma.data.cpu()**2)).sum(dim=1))
      log_psi = -log_psi.view(-1, 1, 1) * compatible_mask.view(
          1, num_states, num_states)
      # import pdb; pdb.set_trace()

      # log message, shape E X K
      log_msg = torch.log(torch.ones(num_edges, num_states) / float(num_states))

      # unary term, shape N X K
      log_phi = torch.softmax(
          feat.permute(0, 2, 3, 1).contiguous().view(num_nodes, -1),
          dim=1).log().data.cpu()

      intermediate_result = log_phi[edge_in].unsqueeze(dim=2) + log_psi
      # import pdb; pdb.set_trace()
      for ii in range(self.max_iter):
        log_msg = (1 - self.damping) * torch.max(
            intermediate_result + torch.mm(msg_adj.data.cpu(),
                                           log_msg).unsqueeze(dim=2),
            dim=1)[0] + self.damping * log_msg
        log_msg = log_msg - torch.logsumexp(log_msg, dim=1, keepdim=True)

      # shape B X K X H X W
      log_prob = log_phi + torch.mm(incidence_mat.data.cpu(), log_msg)
      log_prob = torch.log_softmax(log_prob, dim=1)
      log_prob = log_prob.view(feat.shape[0], feat.shape[2], feat.shape[3],
                               -1).permute(0, 3, 1, 2).contiguous()

      return log_prob.to(feat.device)

  def mean_field(self, feat, img, dist_diff, msg_node):
    if self.training:
      # use GPU
      msg_node = torch.from_numpy(msg_node)
      num_states = feat.shape[1]
      num_nodes = feat.shape[0] * feat.shape[2] * feat.shape[3]
      num_edges = msg_node.shape[0]
      edge_in = msg_node[:, 0]  # shape E X 1
      edge_out = msg_node[:, 1].contiguous()  # shape E X 1

      # unary term, shape N X K
      prob = torch.softmax(
          feat.permute(0, 2, 3, 1).contiguous().view(num_nodes, -1), dim=1)
      unary = -prob.log()

      # pairwise term, shape E X K X K
      img = img.permute(0, 2, 3, 1).reshape(-1, 3)
      intensity_diff = (
          img[msg_node[:, 0], :] - img[msg_node[:, 1], :])**2  # shape E X 3
      compatible_mask = self.compatible_mask[:num_states, :num_states]
      compatible_mask[-1, :] = compatible_mask[-1, :] * 10.0
      compatible_mask[:, -1] = compatible_mask[:, -1] * 10.0
      kernel_1 = (
          self.w1.abs() *
          torch.exp(-(dist_diff.data.cpu() / (2.0 * self.alpha**2)).sum(dim=1) -
                    (intensity_diff / (2.0 * self.beta**2)).sum(dim=1))).view(
                        -1, 1)
      kernel_2 = (
          self.w2.abs() * torch.exp(-(dist_diff.data.cpu() /
                                      (2.0 * self.gamma**2)).sum(dim=1))).view(
                                          -1, 1)

      Q = prob
      for ii in range(self.max_iter):
        kQ = (kernel_1 + kernel_2) * Q[edge_in]  # shape E X K
        agg_Q = unsorted_segment_sum(
            kQ.unsqueeze(dim=0), edge_out,
            num_nodes).squeeze(dim=0)  # shape N X K
        Q = torch.softmax(-agg_Q.mm(compatible_mask) - unary, dim=1)

      log_prob = torch.log(Q).view(feat.shape[0], feat.shape[2], feat.shape[3],
                                   -1).permute(0, 3, 1, 2).contiguous()

      return log_prob
    else:
      # use CPU
      msg_node = torch.from_numpy(msg_node)
      num_states = feat.shape[1]
      num_nodes = feat.shape[0] * feat.shape[2] * feat.shape[3]
      num_edges = msg_node.shape[0]
      edge_in = msg_node[:, 0]  # shape E X 1
      edge_out = msg_node[:, 1].contiguous()  # shape E X 1

      # unary term, shape N X K
      prob = torch.softmax(
          feat.permute(0, 2, 3, 1).contiguous().view(num_nodes, -1), dim=1)
      unary = -prob.log()

      # pairwise term, shape E X K X K
      img = img.permute(0, 2, 3, 1).reshape(-1, 3).data.cpu()
      intensity_diff = (
          img[msg_node[:, 0], :] - img[msg_node[:, 1], :])**2  # shape E X 3
      compatible_mask = self.compatible_mask[:num_states, :num_states]
      compatible_mask[-1, :] = compatible_mask[-1, :] * 10.0
      compatible_mask[:, -1] = compatible_mask[:, -1] * 10.0
      kernel_1 = (self.w1.abs().data.cpu() *
                  torch.exp(-(dist_diff.data.cpu() /
                              (2.0 * self.alpha.data.cpu()**2)).sum(dim=1) -
                            (intensity_diff /
                             (2.0 * self.beta.data.cpu()**2)).sum(dim=1))).view(
                                 -1, 1)
      kernel_2 = (
          self.w2.abs().data.cpu() *
          torch.exp(-(dist_diff.data.cpu() /
                      (2.0 * self.gamma.data.cpu()**2)).sum(dim=1))).view(
                          -1, 1)

      Q = prob.data.cpu()
      for ii in range(self.max_iter):
        kQ = (kernel_1.data.cpu() +
              kernel_2.data.cpu()) * Q[edge_in.data.cpu()]  # shape E X K
        agg_Q = unsorted_segment_sum(
            kQ.unsqueeze(dim=0), edge_out.data.cpu(),
            num_nodes).squeeze(dim=0)  # shape N X K
        Q = torch.softmax(
            -agg_Q.mm(compatible_mask.data.cpu()) - unary.data.cpu(), dim=1)
        # import pdb; pdb.set_trace()

      log_prob = torch.log(Q).view(feat.shape[0], feat.shape[2], feat.shape[3],
                                   -1).permute(0, 3, 1, 2).contiguous()
      # import pdb; pdb.set_trace()
      return log_prob


def grid(H, W):
  num_nodes = H * W
  G1 = nx.grid_2d_graph(H, W)
  node_map = {gg: ii for ii, gg in enumerate(G1.nodes)}
  G2 = nx.relabel_nodes(G1, node_map)
  return G1, G2


def get_msg_graph(G):
  L = nx.line_graph(G.to_directed())

  # remove redundant edges
  redundant_edges = []
  for edge in L.edges():
    if set(edge[0]) == set(edge[1]):
      redundant_edges.append(edge)

  for edge in redundant_edges:
    L.remove_edge(edge[0], edge[1])

  node_list = sorted(L.nodes)

  adj = nx.adjacency_matrix(L, nodelist=node_list)
  return node_list, adj


def scipy_coo_to_pytorch_sp(A):
  idx = torch.from_numpy(np.vstack((A.row, A.col))).long()
  val = torch.from_numpy(A.data).float()
  shape = torch.from_numpy(np.array(A.shape, dtype=np.int32)).long()
  return torch.sparse.FloatTensor(idx, val, torch.Size(shape))


if __name__ == '__main__':
  CRF = ConditionalRandomField()
  batch_size = 1
  height = 256
  width = 512
  num_instance = 20
  img = np.random.randn(batch_size, height, width, 3).astype(np.float32)
  feat = np.random.randn(batch_size, num_instance, height,
                         width).astype(np.float32)

  # construct graph
  G1, G = grid(height, width)
  G_direct = G.to_directed()
  incidence_mat = nx.incidence_matrix(
      G_direct,
      nodelist=sorted(G_direct.nodes),
      edgelist=sorted(G_direct.edges))
  msg_node, msg_adj = get_msg_graph(G)
  pos = np.array([gg for gg in G1.nodes], dtype=np.float32)  # shape N X 2

  # repeat for each image
  msg_node_bat = []
  for ii in range(batch_size):
    msg_node_bat += [np.array(msg_node) + ii * height * width]

  msg_node_bat = np.concatenate(msg_node_bat, axis=0)
  msg_adj_bat = block_diag([msg_adj.tocoo()] * batch_size, format='coo')
  incidence_mat_bat = block_diag(
      [incidence_mat.tocoo()] * batch_size, format='coo')
  msg_adj_bat = scipy_coo_to_pytorch_sp(msg_adj_bat)
  incidence_mat_bat = scipy_coo_to_pytorch_sp(incidence_mat_bat)

  pos = np.tile(pos, (batch_size, 1))  # shape N X 2

  dist_diff = (
      pos[msg_node_bat[:, 0], :] - pos[msg_node_bat[:, 1], :])**2  # shape E X 2
  img = img.reshape(-1, 3)
  intensity_diff = (
      img[msg_node_bat[:, 0], :] - img[msg_node_bat[:, 1], :])**2  # shape E X 3

  feat = torch.from_numpy(feat)
  dist_diff = torch.from_numpy(dist_diff)
  intensity_diff = torch.from_numpy(intensity_diff)

  print(feat.dtype, incidence_mat_bat.dtype, dist_diff.dtype,
        intensity_diff.dtype)

  log_prob = CRF.forward(feat, incidence_mat_bat, dist_diff, intensity_diff,
                         msg_node_bat, msg_adj_bat)
