import time

import numpy as np
import torch
import torch.nn as nn

EPS = float(np.finfo(np.float32).eps)

__all__ = ['NodeGNN','NodeGNN2' , 'MsgGNN', 'NodeGNN_hyunmok_bp']

class NodeGNN(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of NodeGNN """
    super(NodeGNN, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    # message function
    self.msg_func = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # message attention function
    self.att_head = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid()
    ])

    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    # output function
    self.output_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
        xx for xx in [self.msg_func, self.output_func, self.att_head] if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, degree, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    num_edge = msg_node.shape[0]

    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1, 1)  # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1)
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1)

    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    def _prop(state_prev):
      # 1. compute messages
      state_in = state_prev[edge_in, :]  # shape |E| X D
      state_out = state_prev[edge_out, :]  # shape |E| X D
      # node_degree = degree[edge_in].float()
      # 1.1 degree embedding
      # degree_emd = self.node_func()
      # msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |E| X D
      msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))  # shape: |E| X D

      # 2. aggregate message
      scatter_idx = edge_out.view(-1, 1).expand(-1, self.hidden_dim)
      msg_agg = torch.zeros(num_node, self.hidden_dim).to(b.device) # shape: |V| X D
      if self.aggregate_type == 'att':
        att_weight = self.att_head(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))  # shape: |E| X 1
        att_weight = att_weight.exp()
        sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1, 1), att_weight)
        att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1, 1))
        msg_agg = msg_agg.scatter_add_(0, scatter_idx, att_weight * msg)
      elif self.aggregate_type == 'mean':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
        avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_edge).to(b.device))
        msg_agg /= (avg_norm.view(-1, 1) + EPS)
      elif self.aggregate_type == 'add':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
      else:
        raise ValueError("Non-supported aggregation!")
      # 3. update state
      state_new = self.update_func(msg_agg, state_prev)  # GRU update

      return state_new

    # propagation
    norm_loss = 0
    state_hist = []
    for tt in range(self.num_prop):
      state = _prop(state)
      # norm = torch.norm(state.cpu().detach(), 'fro')
      # norm_loss += norm
    # norm_loss = norm_loss / self.num_prop


    # cnt = 1
    # while True:
    #   new_state = _prop(state)
    #   frobenius_norm = torch.norm(new_state - state, p = 'fro')
    #   state = new_state
    #   cnt += 1
    #   if(cnt % 100 == 0):
    #     print("count: ", cnt)
    #   if frobenius_norm  < 0.4:
    #     new_state = new_state.detach()
    #     frobenius_norm = frobenius_norm.detach()
    #     print(frobenius_norm)
    #     break

    # output

    y = self.output_func(torch.cat([state, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss, norm_loss, state_hist

class NodeGNN2(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of NodeGNN """
    super(NodeGNN2, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    # message function
    # self.msg_func = nn.Sequential(*[
    #     nn.Linear(2 * self.hidden_dim + 8, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, self.hidden_dim)
    # ])

    self.msg_func = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 1, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # message attention function
    self.att_head = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid()
    ])

    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    # output function
    self.output_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
        xx for xx in [self.msg_func, self.output_func, self.att_head] if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, degree, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    num_edge = msg_node.shape[0]

    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1, 1)  # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    # ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1)
    # ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1)
    # ff = torch.cat([b[edge_in], b[edge_out], J_msg], dim=1)

    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    def _prop(state_prev):
      # 1. compute messages
      state_in = state_prev[edge_in, :]  # shape |E| X D
      state_out = state_prev[edge_out, :]  # shape |E| X D
      # node_degree = degree[edge_in].float()
      # 1.1 degree embedding
      # degree_emd = self.node_func()
      # msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |E| X D
      # msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))  # shape: |E| X D
      msg = self.msg_func(torch.cat([state_in, state_out, J_msg], dim=1))  # shape: |E| X D
      # 2. aggregate message
      scatter_idx = edge_out.view(-1, 1).expand(-1, self.hidden_dim)
      msg_agg = torch.zeros(num_node, self.hidden_dim).to(b.device) # shape: |V| X D
      if self.aggregate_type == 'att':
        att_weight = self.att_head(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))  # shape: |E| X 1
        att_weight = att_weight.exp()
        sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1, 1), att_weight)
        att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1, 1))
        msg_agg = msg_agg.scatter_add_(0, scatter_idx, att_weight * msg)
      elif self.aggregate_type == 'mean':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
        avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_edge).to(b.device))
        msg_agg /= (avg_norm.view(-1, 1) + EPS)
      elif self.aggregate_type == 'add':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
      else:
        raise ValueError("Non-supported aggregation!")
      # 3. update state
      state_new = self.update_func(msg_agg, state_prev)  # GRU update

      return state_new

    # propagation
    norm_loss = 0
    state_hist = []
    for tt in range(self.num_prop):
      state = _prop(state)
      # norm = torch.norm(state.cpu().detach(), 'fro')
      # norm_loss += norm
    # norm_loss = norm_loss / self.num_prop


    # cnt = 1
    # while True:
    #   new_state = _prop(state)
    #   frobenius_norm = torch.norm(new_state - state, p = 'fro')
    #   state = new_state
    #   cnt += 1
    #   if(cnt % 100 == 0):
    #     print("count: ", cnt)
    #   if frobenius_norm  < 0.4:
    #     new_state = new_state.detach()
    #     frobenius_norm = frobenius_norm.detach()
    #     print(frobenius_norm)
    #     break

    # output

    # y = self.output_func(torch.cat([state, b, -b], dim=1))
    y = self.output_func(state)
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss, norm_loss, state_hist

class MsgGNN_option1(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of MsgGNN """
    super(MsgGNN_option1, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'avg', 'sum'], 'not implemented'

    # message function
    self.msg_func = nn.Sequential(*[
        nn.Linear(2 * self.hidden_dim + 16, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim),
    ])
    # message attention function
    self.att_head = nn.Sequential(*[
        nn.Linear(2 * self.hidden_dim + 16, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim
    )
    # output attention function
    self.graph_output_head_att = nn.Sequential(*[
        nn.Linear(self.hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # output function
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.output_func, self.att_head, self.graph_output_head_att]
      if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2 : (e.g. node 0 -> node 1)
      msg_edge: shape |E| X |E| : (e.g. (row 0: node 0 -> node 1) -> (col 3: node 1 -> node 3) )
      target: shape |V| X 2
    """
    num_node = b.shape[0] # |V|
    num_msg_node = msg_node.shape[0] # |E| = |msgV|
    # idx_msg_edge = torch.nonzero(msg_edge).to(b.device) # |msgE| X 2
    num_msg_edge = idx_msg_edge.shape[0] # |msgE|

    b_in = b[msg_node[:, 0]].view(-1, 1) # shape |E| X 1
    b_out = b[msg_node[:, 1]].view(-1, 1) # shape |E| X 1
    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1, 1) # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    state = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)

    if len(idx_msg_edge.shape) > 1: # in case there are no edges in message graph
      edge_in = idx_msg_edge[:, 0]
      edge_out = idx_msg_edge[:, 1].contiguous()

      ff_in = torch.cat([b_in[edge_in], -b_in[edge_in], b_out[edge_in], -b_out[edge_in],
                         J_msg[edge_in], -J_msg[edge_in], -J_msg[edge_in], J_msg[edge_in]], dim=1)
      ff_out = torch.cat([b_in[edge_out], -b_in[edge_out], b_out[edge_out], -b_out[edge_out],
                          J_msg[edge_out], -J_msg[edge_out], -J_msg[edge_out], J_msg[edge_out]], dim=1)

      def _prop(state_prev):
        state_in = state_prev[edge_in, :]  # shape: |msgE| X D
        state_out = state_prev[edge_out, :]  # shape: |msgE| X D
        # 1. compute messages
        msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |msgE| X D
        # 2. aggregate messages
        scatter_idx = edge_out.view(-1,1).expand(-1, self.hidden_dim) # shape: |msgE| X D
        msg_agg = torch.zeros(num_msg_node, self.hidden_dim).to(b. device)
        if self.aggregate_type == 'att':
          att_weight = self.att_head(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |msgE| X 1
          att_weight = att_weight.exp()
          sftmx_norm = torch.zeros(num_msg_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1,1), att_weight)
          att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1,1))
          msg_agg = msg_agg.scatter_add_(0, scatter_idx, att_weight * msg) # shape: |E| X D
        elif self.aggregate_type == 'avg':
          msg_agg = msg_agg.scatter_add_(0, scatter_idx, msg)  # shape: |E| X D
          avg_norm = torch.zeros(num_msg_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_msg_edge).to(b.device))
          msg_agg /= (avg_norm.view(-1,1) + EPS)  # shape: M X D
        elif self.aggregate_type == 'sum':
          msg_agg = msg_agg.scatter_add_(0, scatter_idx, msg)  # shape: |E| X D
        else:
          raise ValueError("Non-supported aggregation!")
        # 3. update state
        state_new = self.update_func(msg_agg, state_prev)  # GRU update

        return state_new

      # propagation
      for tt in range(self.num_prop):
        state = _prop(state)

    # aggregate states
    out_scatter_idx = msg_node[:, 1].view(-1, 1).expand(-1, self.hidden_dim)
    state_agg = torch.zeros(num_node, self.hidden_dim).to(b.device)
    if self.aggregate_type == 'att':
      out_att_weight = self.graph_output_head_att(state)
      out_att_weight = out_att_weight.exp()
      out_sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, msg_node[:,1].view(-1, 1), out_att_weight)
      out_att_weight = out_att_weight / torch.gather(out_sftmx_norm, 0, msg_node[:,1].view(-1, 1))
      state_agg = state_agg.scatter_add_(0, out_scatter_idx, out_att_weight * state)
    elif self.aggregate_type == 'avg':
      state_agg = state_agg.scatter_add_(0, out_scatter_idx, state)
      out_avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, msg_node[:,1], torch.ones(num_msg_node).to(b.device))
      state_agg /= (out_avg_norm.view(-1,1) + EPS)
    elif self.aggregate_type == 'sum':
      state_agg = state_agg.scatter_add_(0, out_scatter_idx, state)
    else:
      raise ValueError("Non-supported aggregation!")

    # output
    y = self.output_func(torch.cat([state_agg, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss

class MsgGNN(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of MsgGNN """
    super(MsgGNN, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'avg', 'sum'], 'not implemented'

    # message function
    self.msg_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 8, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim),
    ])
    # message attention function
    self.att_head = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim
    )
    # output attention function
    self.graph_output_head_att = nn.Sequential(*[
        nn.Linear(self.hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # output function
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.output_func, self.att_head, self.graph_output_head_att]
      if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edg, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, degree, target=None):
    """
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2 : (e.g. node 0 -> node 1)
      msg_edge: shape |E| X |E| : (e.g. (row 0: node 0 -> node 1) -> (col 3: node 1 -> node 3) )
      target: shape |V| X 2
    """
    num_node = b.shape[0] # |V|
    num_msg_node = msg_node.shape[0] # |E| = |msgV|
    # idx_msg_edge = torch.nonzero(msg_edge).to(b.device)  # |msgE| X 2
    num_msg_edge = idx_msg_edge.shape[0]  # |msgE|

    b_in = b[msg_node[:, 0]].view(-1, 1) # shape |E| X 1
    b_out = b[msg_node[:, 1]].view(-1, 1) # shape |E| X 1
    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1, 1) # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    state = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)

    if len(idx_msg_edge.shape) > 1: # in case there are no edges in message graph
      edge_in = idx_msg_edge[:, 0]
      edge_out = idx_msg_edge[:, 1].contiguous()

      ff_in = torch.cat([b_in[edge_in], -b_in[edge_in], b_out[edge_in], -b_out[edge_in],
                         J_msg[edge_in], -J_msg[edge_in], -J_msg[edge_in], J_msg[edge_in]], dim=1)
      ff_out = torch.cat([b_in, -b_in, b_out, -b_out, J_msg, -J_msg, -J_msg, J_msg], dim=1)

      def _prop(state_prev):
        state_in = state_prev[edge_in, :]  # shape: |msgE| X D
        state_out = state_prev[edge_out, :]  # shape: |msgE| X D
        # 1. aggregate states
        scatter_idx = edge_out.view(-1,1).expand(-1, self.hidden_dim) # shape: |msgE| X D
        state_agg = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)
        if self.aggregate_type == 'att':
          att_weight = self.att_head(torch.cat([state_in, ff_in], dim=1)) # shape: |msgE| X 1
          att_weight = att_weight.exp()
          sftmx_norm = torch.zeros(num_msg_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1,1), att_weight)
          att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1,1))
          state_agg = state_agg.scatter_add_(0, scatter_idx, att_weight * state_in) # shape: |E| X D
        elif self.aggregate_type == 'avg':
          state_agg = state_agg.scatter_add_(0, scatter_idx, state_in)  # shape: |E| X D
          avg_norm = torch.zeros(num_msg_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_msg_edge).to(b.device))
          state_agg /= (avg_norm.view(-1,1) + EPS)  # shape: M X D
        elif self.aggregate_type == 'sum':
          state_agg = state_agg.scatter_add_(0, scatter_idx, state_in)  # shape: |E| X D
        else:
          raise ValueError("Non-supported aggregation!")
        # 2. compute messages
        msg = self.msg_func(torch.cat([state_agg, ff_out], dim=1))  # shape: |msgE| X D
        # 3. update state
        state_new = self.update_func(msg, state_prev)  # GRU update

        return state_new

      # propagation
      for tt in range(self.num_prop):
        state = _prop(state)

    # aggregate states
    out_scatter_idx = msg_node[:, 1].view(-1, 1).expand(-1, self.hidden_dim)
    out_state_agg = torch.zeros(num_node, self.hidden_dim).to(b.device)
    if self.aggregate_type == 'att':
      out_att_weight = self.graph_output_head_att(state)
      out_att_weight = out_att_weight.exp()
      out_sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, msg_node[:,1].view(-1, 1), out_att_weight)
      out_att_weight = out_att_weight / torch.gather(out_sftmx_norm, 0, msg_node[:,1].view(-1, 1))
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, out_att_weight * state)
    elif self.aggregate_type == 'avg':
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
      out_avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, msg_node[:,1], torch.ones(num_msg_node).to(b.device))
      out_state_agg /= (out_avg_norm.view(-1,1) + EPS)
    elif self.aggregate_type == 'sum':
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
    else:
      raise ValueError("Non-supported aggregation!")

    # output
    y = self.output_func(torch.cat([out_state_agg, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss, 0, 0

class MsgGNN_option3(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of MsgGNN """
    super(MsgGNN_option3, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'avg', 'sum'], 'not implemented'

    # message function
    self.msg_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 8, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim),
        nn.Softmax(dim=1)
    ])
    # message attention function
    self.att_head = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim
    )
    # update softmax
    self.update_softmax = nn.Softmax(dim=1)
    # output attention function
    self.graph_output_head_att = nn.Sequential(*[
        nn.Linear(self.hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ])
    # output function
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.output_func, self.att_head, self.graph_output_head_att]
      if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2 : (e.g. node 0 -> node 1)
      msg_edge: shape |E| X |E| : (e.g. (row 0: node 0 -> node 1) -> (col 3: node 1 -> node 3) )
      target: shape |V| X 2
    """
    num_node = b.shape[0] # |V|
    num_msg_node = msg_node.shape[0] # |E| = |msgV|
    # idx_msg_edge = torch.nonzero(msg_edge).to(b.device) # |msgE| X 2
    num_msg_edge = idx_msg_edge.shape[0] # |msgE|

    b_in = b[msg_node[:, 0]].view(-1, 1) # shape |E| X 1
    b_out = b[msg_node[:, 1]].view(-1, 1) # shape |E| X 1
    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1,1) # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    state = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)

    if len(idx_msg_edge.shape) > 1: # in case there are no edges in message graph
      edge_in = idx_msg_edge[:, 0]
      edge_out = idx_msg_edge[:, 1].contiguous()

      ff_in = torch.cat([b_in[edge_in], -b_in[edge_in], b_out[edge_in], -b_out[edge_in],
                         J_msg[edge_in], -J_msg[edge_in], -J_msg[edge_in], J_msg[edge_in]], dim=1)
      ff_out = torch.cat([b_in, -b_in, b_out, -b_out, J_msg, -J_msg, -J_msg, J_msg], dim=1)

      def _prop(state_prev):
        state_in = state_prev[edge_in, :]  # shape: |msgE| X D
        state_out = state_prev[edge_out, :]  # shape: |msgE| X D
        # 1. aggregate states
        scatter_idx = edge_out.view(-1,1).expand(-1, self.hidden_dim) # shape: |msgE| X D
        state_agg = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)
        if self.aggregate_type == 'att':
          att_weight = self.att_head(torch.cat([state_in, ff_in], dim=1)) # shape: |msgE| X 1
          att_weight = att_weight.exp()
          sftmx_norm = torch.zeros(num_msg_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1,1), att_weight)
          att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1,1))
          state_agg = state_agg.scatter_add_(0, scatter_idx, att_weight * state_in) # shape: |E| X D
        elif self.aggregate_type == 'avg':
          state_agg = state_agg.scatter_add_(0, scatter_idx, state_in)  # shape: |E| X D
          avg_norm = torch.zeros(num_msg_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_msg_edge).to(b.device))
          state_agg /= (avg_norm.view(-1,1) + EPS)  # shape: M X D
        elif self.aggregate_type == 'sum':
          state_agg = state_agg.scatter_add_(0, scatter_idx, state_in)  # shape: |E| X D
        else:
          raise ValueError("Non-supported aggregation!")
        # 2. compute messages
        msg = self.msg_func(torch.cat([state_agg, ff_out], dim=1))  # shape: |msgE| X D
        # 3. update state
        state_new = self.update_func(msg, state_prev)  # GRU update
        # 4. softmax
        state_new = self.update_softmax(state_new)

        return state_new

      # propagation
      for tt in range(self.num_prop):
        state = _prop(state)

    # aggregate states
    out_scatter_idx = msg_node[:, 1].view(-1, 1).expand(-1, self.hidden_dim)
    out_state_agg = torch.zeros(num_node, self.hidden_dim).to(b.device)
    if self.aggregate_type == 'att':
      out_att_weight = self.graph_output_head_att(state)
      out_att_weight = out_att_weight.exp()
      out_sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, msg_node[:,1].view(-1, 1), out_att_weight)
      out_att_weight = out_att_weight / torch.gather(out_sftmx_norm, 0, msg_node[:,1].view(-1, 1))
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, out_att_weight * state)
    elif self.aggregate_type == 'avg':
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
      out_avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, msg_node[:,1], torch.ones(num_msg_node).to(b.device))
      out_state_agg /= (out_avg_norm.view(-1,1) + EPS)
    elif self.aggregate_type == 'sum':
      out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
    else:
      raise ValueError("Non-supported aggregation!")

    # output
    y = self.output_func(torch.cat([out_state_agg, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss

class NodeGNN_hyunmok_bp(nn.Module):

  def __init__(self, config):
    """ A simplified implementation of NodeGNN """
    super(NodeGNN_hyunmok_bp, self).__init__()
    self.config = config
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    assert self.aggregate_type in ['att', 'avg', 'sum'], 'not implemented'

    # message function
    self.msg_func = nn.Sequential(*[
        nn.Linear(2 * self.hidden_dim + 8, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim)
    ])
    # message attention function
    self.att_head = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid()
    ])
    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim, hidden_size=self.hidden_dim)
    # output function
    self.output_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ])

    if self.config.model.loss == 'KL-pq' or self.config.model.loss == 'KL-qp':
      self.loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
      self.loss_func = nn.MSELoss(reduction='mean')

    self._init_param()

  def _init_param(self):
    mlp_modules = [
        xx for xx in [self.msg_func, self.output_func, self.att_head] if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  # def forward(self, J, b, msg_node, msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    num_edge = msg_node.shape[0]

    # J_msg = J[msg_node[:, 0], msg_node[:, 1]].view(-1, 1)  # shape |E| X 1
    # J_msg = torch.Tensor([J[msg_node[r, 0], msg_node[r, 1]] for r in range(msg_node.shape[0])]).view(-1, 1)

    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1)
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1)

    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    def _prop(state_prev):
      # 1. compute messages
      state_in = state_prev[edge_in, :]  # shape |E| X D
      state_out = state_prev[edge_out, :]  # shape |E| X D
      msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |E| X D
      # 2. aggregate message
      scatter_idx = edge_out.view(-1, 1).expand(-1, self.hidden_dim)
      msg_agg = torch.zeros(num_node, self.hidden_dim).to(b.device) # shape: |V| X D
      if self.aggregate_type == 'att':
        att_weight = self.att_head(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))  # shape: |E| X 1
        att_weight = att_weight.exp()
        sftmx_norm = torch.zeros(num_node, 1).to(b.device).scatter_add_(0, edge_out.view(-1, 1), att_weight)
        att_weight = att_weight / torch.gather(sftmx_norm, 0, edge_out.view(-1, 1))
        msg_agg = msg_agg.scatter_add_(0, scatter_idx, att_weight * msg)
      elif self.aggregate_type == 'avg':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
        avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_edge).to(b.device))
        msg_agg /= (avg_norm.view(-1, 1) + EPS)
      elif self.aggregate_type == 'sum':
        msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
      else:
        raise ValueError("Non-supported aggregation!")
      # 3. update state
      state_new = self.update_func(msg_agg, state_prev)  # GRU update

      return state_new

    # propagation
    loss = 0
    y_step = []
    for tt in range(self.num_prop):
      state = _prop(state)

      y = self.output_func(torch.cat([state, b, -b], dim=1))
      y = torch.log_softmax(y, dim=1)
      y_step += [np.expand_dims(torch.exp(y[:, 0]).data.cpu().numpy(), axis = 0)]

      if target is not None:
        if self.config.model.loss == 'KL-pq':
          # criterion(logQ, P) = KL(P||Q)
          loss = self.loss_func(y, torch.cat([torch.unsqueeze(target[:, tt], 0),
                                              torch.unsqueeze(1-target[:, tt], 0)], dim = 0).transpose(0,1))
        elif self.config.model.loss == 'KL-qp':
          # criterion(logP, Q) = KL(Q||P)
          loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
        elif self.config.model.loss == 'MSE':
          loss = self.loss_func(y, torch.log(target))
        else:
          raise ValueError("Non-supported loss function!")
      else:
        loss = None
      loss += loss

    y_step = np.concatenate(y_step, axis = 0).transpose()
    return y_step, loss



