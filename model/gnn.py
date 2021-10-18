import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

EPS = float(np.finfo(np.float32).eps)

__all__ = ['TorchGNN', 'TorchGNN_meta', 'TorchGNN_meta_edge', 'TorchGNN_multitask', 'TorchGNN_MsgGNN', 'TorchGNN_MsgGNN_parallel']

class TorchGNN(nn.Module):
  def __init__(self, config, test=False):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.batch_size = 1 if test else config.train.batch_size
    self.masking = config.model.masking if config.model.masking is not None else False
    self.SSL = config.model.SSL if config.model.SSL is not None else False
    self.train_pretext = config.model.train_pretext if config.model.train_pretext is not None else False
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      print("Using attention")
      self.GeoLayer = MYGATConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      # nn.Dropout(p=0.1),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 64),
      # nn.Dropout(p=0.5),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    #PRETEXT
    if self.SSL:
      self.pretext_output_func = nn.Sequential(*[
        nn.Linear(self.hidden_dim + 2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
      ])

    #JUMPING KNOWLEDGE
    self.output_JK = nn.Sequential(*[
      nn.Linear(self.hidden_dim * self.num_prop,  64),
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
      xx for xx in [self.output_func] if xx is not None
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

  # def forward(self, J_msg, b, msg_node, degree, idx_msg_edge, target=None):
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
    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T)

      if self.jumping:
        hidden_state.append(state)

    #jumping knowledge
    if self.jumping:
      state = torch.cat(hidden_state, dim=-1)
      y = self.output_JK(state)
    else:
      if self.SSL:
        if self.train_pretext:
          y = self.pretext_output_func(torch.cat([state, b, -b], dim=1))  #READOUT
        else:
          y = self.output_func(torch.cat([state, b, -b], dim=1))  # READOUT
      else:
        y = self.output_func(torch.cat([state, b, -b], dim=1)) #READOUT

    if not self.SSL:
      y = torch.log_softmax(y, dim=1)
    elif self.SSL and not self.train_pretext:
      y = torch.log_softmax(y, dim=1)

    #############
    # Master node
    #############
    if self.master_node:
      num_node = self.num_node
      # cal_list = torch.tensor([i+j*(num_node+1) for j in range(self.batch_size) for i in range(num_node)]).to(b.device)
      cal_list = torch.tensor([i+2*num_node*j for j in range(self.batch_size) for i in range(num_node)]).to(b.device)
      y = torch.index_select(y, 0, cal_list)

    #############
    # Masking
    #############
    if self.masking:
      # cal_list = [i for i in range(self.num_node * self.batch_size) if len(np.where(msg_node.cpu()[1, :] == i)[0])<self.config.model.masking_number]
      cal_list = torch.tensor([i for i in range(self.num_node * self.batch_size) if len(np.where(msg_node.cpu()[1, :] == i)[0]) in self.config.model.masking_number]).to(b.device)
      y = torch.index_select(y, 0, cal_list)
      # y = y[cal_list]
      target = torch.index_select(target, 0, cal_list)
      # target = target[cal_list]

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        if self.SSL:
          if self.train_pretext:
            loss = self.loss_func(y, target)
        else:
          loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    return y, loss

class TorchGNN_meta(nn.Module):
  def __init__(self, config, test=False):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN_meta, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.batch_size = config.test.batch_size if test else config.train.batch_size
    self.masking = config.model.masking if config.model.masking is not None else False
    self.SSL = config.model.SSL if config.model.SSL is not None else False
    self.train_pretext = config.model.train_pretext if config.model.train_pretext is not None else False
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      print("Using attention")
      self.GeoLayer = MYGATConv_meta(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer_meta(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
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
      xx for xx in [self.output_func] if xx is not None
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

  # def forward(self, J_msg, b, msg_node, degree, idx_msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, target=None, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

      if self.jumping:
        hidden_state.append(state)


    y = self.output_func(torch.cat([state, b, -b], dim=1)) #READOUT
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        if self.SSL:
          if self.train_pretext:
            loss = self.loss_func(y, target)
        else:
          loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    y_, target_ = y.view(-1, self.num_node, 2), target.view(-1, self.num_node, 2)
    loss_per_batch = [self.loss_func(_, __) for _, __ in zip(y_, target_)]
    return y, loss, loss_per_batch

class TorchGNN_meta_edge(nn.Module):
  def __init__(self, config, test=False):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN_meta_edge, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.batch_size = config.test.batch_size if test else config.train.batch_size
    self.masking = config.model.masking if config.model.masking is not None else False
    self.SSL = config.model.SSL if config.model.SSL is not None else False
    self.train_pretext = config.model.train_pretext if config.model.train_pretext is not None else False
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      print("Using attention")
      self.GeoLayer = MYGATConv_meta_edge(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer_meta(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
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
      xx for xx in [self.output_func] if xx is not None
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

  # def forward(self, J_msg, b, msg_node, degree, idx_msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, target=None, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

      if self.jumping:
        hidden_state.append(state)


    y = self.output_func(torch.cat([state, b, -b], dim=1)) #READOUT
    y = torch.log_softmax(y, dim=1)

    if target is not None:
      if self.config.model.loss == 'KL-pq':
        # criterion(logQ, P) = KL(P||Q)
        loss = self.loss_func(y, target)
      elif self.config.model.loss == 'KL-qp':
        # criterion(logP, Q) = KL(Q||P)
        loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
      elif self.config.model.loss == 'MSE':
        if self.SSL:
          if self.train_pretext:
            loss = self.loss_func(y, target)
        else:
          loss = self.loss_func(y, torch.log(target))
      else:
        raise ValueError("Non-supported loss function!")
    else:
      loss = None

    y_, target_ = y.view(-1, self.num_node, 2), target.view(-1, self.num_node, 2)
    loss_per_batch = [self.loss_func(_, __) for _, __ in zip(y_, target_)]
    return y, loss, loss_per_batch

class TorchGNN_multitask(nn.Module):
  def __init__(self, config, test=False):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN_multitask, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.batch_size = 1 if test else config.train.batch_size
    self.masking = config.model.masking if config.model.masking is not None else False
    self.SSL = config.model.SSL if config.model.SSL is not None else False
    self.train_pretext = config.model.train_pretext if config.model.train_pretext is not None else False
    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      print("Using attention")
      self.GeoLayer = MYGATConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    #MULTITASK
    self.multitask_output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      # nn.Linear(64, 1)
      nn.Linear(64, 300),
      # nn.Sigmoid()
    ])

    self.loss_func1 = nn.KLDivLoss(reduction='batchmean')
    self.loss_func2 = nn.MSELoss(reduction='mean')
    self.loss_func3 = nn.CrossEntropyLoss(reduction='mean')
    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.output_func, self.multitask_output_func] if xx is not None
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

  # def forward(self, J_msg, b, msg_node, degree, idx_msg_edge, target=None):
  def forward(self, J_msg, b, msg_node, idx_msg_edge, task_index, target=None):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    num_node = b.shape[0]
    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T)

      if self.jumping:
        hidden_state.append(state)
    if task_index == 0:
      y = self.output_func(torch.cat([state, b, -b], dim=1))  # READOUT
      y = torch.log_softmax(y, dim=1)
      loss = self.loss_func1(y, target[0])
    elif task_index == 1:
      y = self.multitask_output_func(torch.cat([state, b, -b], dim=1))  #READOUT
      # loss = self.loss_func2(torch.log10(y), torch.log10(target[1].view(-1, 1))) #MSE_LOGSCALE

      loss = self.loss_func2(y, target[1]) #MSE_PATTERN_TREE
      # loss = self.loss_func1(torch.softmax(y, dim=1), torch.softmax(target[1], dim=1)) #Cross_PATTERN_TREE

      # y = torch.round(y*99)  #ONEHOT
      # y = torch.nn.functional.one_hot(y.long(), num_classes=100).view(-1,100).float()
      # loss = self.loss_func3(y, target[1].view(-1).long())
    return y, loss

class GeoLayer(MessagePassing):
  def __init__(self, hidden_dim, degree_emb, aggre_type, skip_connection=False, interpol=False):
    super(GeoLayer, self).__init__(aggr=aggre_type)  # "Add" aggregation (Step 5).
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol

    # message function
    if self.degree_emb:
      c = 3
    else:
      c = 2

    self.msg_func = nn.Sequential(*[
      nn.Linear(c * self.hidden_dim + 8, 64),
      # nn.Dropout(p=0.1),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 64),
      # nn.Dropout(p=0.5),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # degree embedding
    if self.degree_emb is True:
      self.degree_func = nn.Sequential(*[
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim)
      ])
    else:
      self.degree_func = None

    # update function
    self.update_func = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_down = nn.Sequential(*[
      nn.Linear(2*self.hidden_dim, self.hidden_dim),
      nn.ReLU(),
      #nn.Linear(self.hidden_dim, self.hidden_dim),
      #nn.ReLU()
    ])

    self.gating = nn.Sequential(*[
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.Sigmoid()
    ])

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.degree_func, self.update] if xx is not None
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

  # def forward(self, msg_node, J_msg, b, state_prev, degree, idx_msg_edge):
  def forward(self, msg_node, J_msg, b, state_prev, idx_msg_edge):
    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1).float()
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1).float()

    # ff = torch.cat([J_msg, b[edge_in], b[edge_out]], dim=1).float()

    state_in = state_prev[edge_in, :]  # shape |E| X D
    state_out = state_prev[edge_out, :]  # shape |E| X D
    # node_degree = degree[edge_out].float()
    # node_degree = self.degree_func(degree[edge_out].view(-1,1).float()) if self.degree_emb is True else 0
    return self.propagate(edge_index = msg_node.t(), state_prev=state_prev, J_msg=J_msg, state_in=state_in,
                          state_out=state_out,
                          ff_in=ff_in, ff_out=ff_out)#, node_degree=node_degree)

  # def message(self, state_in, state_out, node_degree, ff_in, ff_out):
  def message(self, state_in, state_out, ff_in, ff_out):
    if self.degree_emb:
      out = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out, node_degree], dim=1))
      # out = self.msg_func(torch.cat([state_in, state_out, ff, node_degree], dim=1))
    else:
      out = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))
      # out = self.msg_func(torch.cat([state_in, state_out, ff], dim=1))
    return out

  def update(self, msg_agg, state_prev):
    out = self.update_func(msg_agg, state_prev)
    #skip-connection
    if self.skip_connection:
      if self.interpol:
        # skip-connection(interpol)
        alpha = self.gating(state_prev)
        out = alpha * out + (1-alpha) * state_prev
      else:
        # skip-connection
        out = torch.cat([out, state_prev], dim=-1)
        out = self.update_down(out)
    return out

class GeoLayer_meta(MessagePassing):
  def __init__(self, hidden_dim, degree_emb, aggre_type, skip_connection=False, interpol=False):
    super(GeoLayer_meta, self).__init__(aggr=aggre_type)  # "Add" aggregation (Step 5).
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol


    self.msg_func = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # update function
    self.update_func1 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_func2 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.update_func1, self.update_func2] if xx is not None
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

  # def forward(self, msg_node, J_msg, b, state_prev, degree, idx_msg_edge):
  def forward(self, msg_node, J_msg, b, state_prev, idx_msg_edge, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1).float()
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1).float()

    state_in = state_prev[edge_in, :]  # shape |E| X D
    state_out = state_prev[edge_out, :]  # shape |E| X D
    return self.propagate(edge_index = msg_node.t(), state_prev=state_prev, J_msg=J_msg, state_in=state_in,
                          state_out=state_out,
                          ff_in=ff_in, ff_out=ff_out, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

  # def message(self, state_in, state_out, node_degree, ff_in, ff_out):
  def message(self, state_in, state_out, ff_in, ff_out, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    out = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))

    # out = torch.zeros(state_in.size(0), self.hidden_dim).cuda()
    # for i in range(len(edge_idx)):
    #   if len(edge_idx[i]) == 0: continue
    #   input = torch.cat(
    #     [state_in[edge_idx[i], :], ff_in[edge_idx[i], :], state_out[edge_idx[i], :], ff_out[edge_idx[i], :]], dim=1)
    #   if i == 0:
    #     aux = self.msg_func1(input)
    #   else:
    #     aux = self.msg_func2(input)
    #   out[edge_idx[i], :] = aux

    return out

  def update(self, msg_agg, state_prev, node_idx, node_idx_inv):
    out = torch.zeros(msg_agg.size(0), self.hidden_dim).cuda()
    for i in range(len(node_idx)):
      if len(node_idx[i]) == 0: continue
      msg = msg_agg[node_idx[i], :]
      state = state_prev[node_idx[i], :]
      # Updates node's value; no worries about updating too early since each node only affects itself.
      if i==0:
        aux = self.update_func1(msg, state)
      else:
        aux = self.update_func2(msg, state)
      out[node_idx[i], :] = aux
    return out

class MYGATConv(GATConv):
  def __init__(self, in_channels, out_channels, hidden_dim, degree_emb, skip_connection, interpol):
    super(MYGATConv, self).__init__(in_channels=in_channels, out_channels=out_channels, add_self_loops=False)
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol

    # message function
    c = 3 if self.degree_emb == True else 2
    self.msg_func = nn.Sequential(*[
      nn.Linear(c * self.hidden_dim + 8, 64),
      # nn.Dropout(p=0.1),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 64),
      # nn.Dropout(p=0.5),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # degree embedding
    if self.degree_emb == True:
      self.degree_func = nn.Sequential(*[
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim)
      ])
    else:
      self.degree_func = None

    # update function
    self.update_func = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_down = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim, self.hidden_dim),
      nn.ReLU()
    ])

    self.gating = nn.Sequential(*[
      nn.Linear(self.hidden_dim, 1),
      nn.Sigmoid()
    ])

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.degree_func, self.update, self.update_down, self.gating] if xx is not None
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

  # def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, degree, idx_msg_edge, size: Size = None, return_attention_weights=None):
  def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, idx_msg_edge,
                size: Size = None, return_attention_weights=None):
    edge_in = edge_index[:, 0]
    edge_out = edge_index[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1).float()
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1).float()

    state_in = x[edge_in, :]  # shape |E| X D
    state_out = x[edge_out, :]  # shape |E| X D

    H, C = self.heads, self.out_channels

    x_l: OptTensor = None
    x_r: OptTensor = None
    alpha_l: OptTensor = None
    alpha_r: OptTensor = None
    if isinstance(x, Tensor):
      assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = x_r = self.lin_l(x).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      alpha_r = (x_r * self.att_r).sum(dim=-1)
    else:
      x_l, x_r = x[0], x[1]
      assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = self.lin_l(x_l).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      if x_r is not None:
        x_r = self.lin_r(x_r).view(-1, H, C)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

    assert x_l is not None
    assert alpha_l is not None

    if self.add_self_loops:
      if isinstance(edge_index, Tensor):
        num_nodes = x_l.size(0)
        if x_r is not None:
          num_nodes = min(num_nodes, x_r.size(0))
        if size is not None:
          num_nodes = min(size[0], size[1])
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
      elif isinstance(edge_index, SparseTensor):
        edge_index = set_diag(edge_index)
    # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
    out = self.propagate(edge_index.t(), x=(x_l, x_r),
                         alpha=(alpha_l, alpha_r), size=size, state_in=state_in, ff_in=ff_in, state_out=state_out, ff_out=ff_out, state_prev = x)
                        #,node_degree = node_degree)

    alpha = self._alpha
    self._alpha = None

    if self.concat:
      out = out.view(-1, self.heads * self.out_channels)
    else:
      out = out.mean(dim=1)

    if self.bias is not None:
      out += self.bias

    if isinstance(return_attention_weights, bool):
      assert alpha is not None
      if isinstance(edge_index, Tensor):
        return out, (edge_index, alpha)
      elif isinstance(edge_index, SparseTensor):
        return out, edge_index.set_value(alpha, layout='coo')
    else:
      return out

  def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
              index: Tensor, ptr: OptTensor,
              size_i: Optional[int], state_in, ff_in, state_out, ff_out)-> Tensor: #, node_degree) -> Tensor:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr, size_i)
    self._alpha = alpha
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    if self.degree_emb == 1:
      x_j = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out, node_degree], dim=1))
    else:
      x_j = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))
    return x_j * alpha

  def update(self, msg_agg, state_prev):
    out = self.update_func(msg_agg, state_prev)
    # skip-connection
    if self.skip_connection:
      if self.interpol:
        # skip-connection(interpol)
        alpha_inter = self.gating(state_prev)
        out = alpha_inter * out + (1 - alpha_inter) * state_prev
      else:
        # skip-connection
        out = torch.cat([out, state_prev], dim=-1)
        out = self.update_down(out)
    return out

class MYGATConv_meta(GATConv):
  def __init__(self, in_channels, out_channels, hidden_dim, degree_emb, skip_connection, interpol):
    super(MYGATConv_meta, self).__init__(in_channels=in_channels, out_channels=out_channels, add_self_loops=False)
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol

    # message function
    self.msg_func = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # update function
    self.update_func1 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_func2 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.gating = nn.Sequential(*[
      nn.Linear(self.hidden_dim, 1),
      nn.Sigmoid()
    ])

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.update_func1, self.update_func2, self.gating] if xx is not None
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

  # def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, degree, idx_msg_edge, size: Size = None, return_attention_weights=None):
  def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, idx_msg_edge,
                size: Size = None, return_attention_weights=None, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    edge_in = edge_index[:, 0]
    edge_out = edge_index[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1).float()
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1).float()

    state_in = x[edge_in, :]  # shape |E| X D
    state_out = x[edge_out, :]  # shape |E| X D

    H, C = self.heads, self.out_channels

    x_l: OptTensor = None
    x_r: OptTensor = None
    alpha_l: OptTensor = None
    alpha_r: OptTensor = None
    if isinstance(x, Tensor):
      assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = x_r = self.lin_l(x).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      alpha_r = (x_r * self.att_r).sum(dim=-1)
    else:
      x_l, x_r = x[0], x[1]
      assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = self.lin_l(x_l).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      if x_r is not None:
        x_r = self.lin_r(x_r).view(-1, H, C)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

    assert x_l is not None
    assert alpha_l is not None

    if self.add_self_loops:
      if isinstance(edge_index, Tensor):
        num_nodes = x_l.size(0)
        if x_r is not None:
          num_nodes = min(num_nodes, x_r.size(0))
        if size is not None:
          num_nodes = min(size[0], size[1])
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
      elif isinstance(edge_index, SparseTensor):
        edge_index = set_diag(edge_index)
    # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
    out = self.propagate(edge_index.t(), x=(x_l, x_r),
                         alpha=(alpha_l, alpha_r), size=size, state_in=state_in, ff_in=ff_in, state_out=state_out, ff_out=ff_out, state_prev = x, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)


    alpha = self._alpha
    self._alpha = None

    if self.concat:
      out = out.view(-1, self.heads * self.out_channels)
    else:
      out = out.mean(dim=1)

    if self.bias is not None:
      out += self.bias

    if isinstance(return_attention_weights, bool):
      assert alpha is not None
      if isinstance(edge_index, Tensor):
        return out, (edge_index, alpha)
      elif isinstance(edge_index, SparseTensor):
        return out, edge_index.set_value(alpha, layout='coo')
    else:
      return out

  def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
              index: Tensor, ptr: OptTensor,
              size_i: Optional[int], state_in, ff_in, state_out, ff_out, edge_idx, edge_idx_inv)-> Tensor: #, node_degree) -> Tensor:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr, size_i)
    self._alpha = alpha
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
    x_j = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))

    # out = torch.zeros(state_in.size(0), self.hidden_dim).cuda()
    # for i in range(len(edge_idx)):
    #   if len(edge_idx[i]) == 0: continue
    #   input = torch.cat([state_in[edge_idx[i], :], ff_in[edge_idx[i], :], state_out[edge_idx[i], :], ff_out[edge_idx[i], :]], dim=1)
    #   if i==0:
    #     aux = self.msg_func1(input)
    #   else:
    #     aux = self.msg_func2(input)
    #   out[edge_idx[i], :] = aux
    # return out * alpha
    return x_j * alpha

  def update(self, msg_agg, state_prev, node_idx, node_idx_inv):
    out = torch.zeros(msg_agg.size(0), self.hidden_dim).cuda()
    for i in range(len(node_idx)):
      if len(node_idx[i]) == 0: continue
      msg = msg_agg[node_idx[i], :]
      state = state_prev[node_idx[i], :]
      # Updates node's value; no worries about updating too early since each node only affects itself.
      if i == 0:
        aux = self.update_func1(msg, state)
      else:
        aux = self.update_func2(msg, state)
      out[node_idx[i], :] = aux
    return out

class MYGATConv_meta_edge(GATConv):
  def __init__(self, in_channels, out_channels, hidden_dim, degree_emb, skip_connection, interpol):
    super(MYGATConv_meta_edge, self).__init__(in_channels=in_channels, out_channels=out_channels, add_self_loops=False)
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol

    # message function
    self.msg_func1 = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    self.msg_func2 = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # update function
    self.update_func1 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_func2 = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.gating = nn.Sequential(*[
      nn.Linear(self.hidden_dim, 1),
      nn.Sigmoid()
    ])

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func1, self.msg_func2, self.update_func1, self.update_func2, self.gating] if xx is not None
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

  # def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, degree, idx_msg_edge, size: Size = None, return_attention_weights=None):
  def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, J_msg, b, idx_msg_edge,
                size: Size = None, return_attention_weights=None, node_idx=None, node_idx_inv=None, edge_idx=None, edge_idx_inv=None):
    edge_in = edge_index[:, 0]
    edge_out = edge_index[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1).float()
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1).float()

    state_in = x[edge_in, :]  # shape |E| X D
    state_out = x[edge_out, :]  # shape |E| X D

    H, C = self.heads, self.out_channels

    x_l: OptTensor = None
    x_r: OptTensor = None
    alpha_l: OptTensor = None
    alpha_r: OptTensor = None
    if isinstance(x, Tensor):
      assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = x_r = self.lin_l(x).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      alpha_r = (x_r * self.att_r).sum(dim=-1)
    else:
      x_l, x_r = x[0], x[1]
      assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
      x_l = self.lin_l(x_l).view(-1, H, C)
      alpha_l = (x_l * self.att_l).sum(dim=-1)
      if x_r is not None:
        x_r = self.lin_r(x_r).view(-1, H, C)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

    assert x_l is not None
    assert alpha_l is not None

    if self.add_self_loops:
      if isinstance(edge_index, Tensor):
        num_nodes = x_l.size(0)
        if x_r is not None:
          num_nodes = min(num_nodes, x_r.size(0))
        if size is not None:
          num_nodes = min(size[0], size[1])
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
      elif isinstance(edge_index, SparseTensor):
        edge_index = set_diag(edge_index)
    # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
    out = self.propagate(edge_index.t(), x=(x_l, x_r),
                         alpha=(alpha_l, alpha_r), size=size, state_in=state_in, ff_in=ff_in, state_out=state_out, ff_out=ff_out, state_prev = x, node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)


    alpha = self._alpha
    self._alpha = None

    if self.concat:
      out = out.view(-1, self.heads * self.out_channels)
    else:
      out = out.mean(dim=1)

    if self.bias is not None:
      out += self.bias

    if isinstance(return_attention_weights, bool):
      assert alpha is not None
      if isinstance(edge_index, Tensor):
        return out, (edge_index, alpha)
      elif isinstance(edge_index, SparseTensor):
        return out, edge_index.set_value(alpha, layout='coo')
    else:
      return out

  def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
              index: Tensor, ptr: OptTensor,
              size_i: Optional[int], state_in, ff_in, state_out, ff_out, edge_idx, edge_idx_inv)-> Tensor: #, node_degree) -> Tensor:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, self.negative_slope)
    alpha = softmax(alpha, index, ptr, size_i)
    self._alpha = alpha
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    out = torch.zeros(state_in.size(0), self.hidden_dim).cuda()
    for i in range(len(edge_idx)):
      if len(edge_idx[i]) == 0: continue
      input = torch.cat([state_in[edge_idx[i], :], ff_in[edge_idx[i], :], state_out[edge_idx[i], :], ff_out[edge_idx[i], :]], dim=1)
      if i==0:
        aux = self.msg_func1(input)
      else:
        aux = self.msg_func2(input)
      out[edge_idx[i], :] = aux
    return out * alpha

  def update(self, msg_agg, state_prev, node_idx, node_idx_inv):
    out = torch.zeros(msg_agg.size(0), self.hidden_dim).cuda()
    for i in range(len(node_idx)):
      if len(node_idx[i]) == 0: continue
      msg = msg_agg[node_idx[i], :]
      state = state_prev[node_idx[i], :]
      # Updates node's value; no worries about updating too early since each node only affects itself.
      if i == 0:
        aux = self.update_func1(msg, state)
      else:
        aux = self.update_func2(msg, state)
      out[node_idx[i], :] = aux
    return out

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

class TorchGNN_MsgGNN(nn.Module):
  def __init__(self, config):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN_MsgGNN, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol

    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      self.GeoLayer = MYGATConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer_msg(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    #JUMPING KNOWLEDGE
    self.output_JK = nn.Sequential(*[
      nn.Linear(self.hidden_dim * self.num_prop,  64),
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
      xx for xx in [self.output_func, self.output_JK] if xx is not None
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
    num_msg_node = msg_node.shape[1]
    # num_msg_edge = idx_msg_edge.shape[1]
    state = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T)

      if self.jumping:
        hidden_state.append(state)

    out_scatter_idx = msg_node.T[:, 1].view(-1, 1).expand(-1, self.hidden_dim)
    out_state_agg = torch.zeros(num_node, self.hidden_dim).to(b.device)
    out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
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

class TorchGNN_MsgGNN_parallel(nn.Module):
  def __init__(self, config):
    """ A simplified implementation of NodeGNN """
    super(TorchGNN_MsgGNN_parallel, self).__init__()
    self.config = config
    self.drop_prob = config.model.drop_prob
    self.num_node = config.dataset.num_node
    self.hidden_dim = config.model.hidden_dim
    self.num_prop = config.model.num_prop
    self.aggregate_type = config.model.aggregate_type
    self.degree_emb = config.model.degree_emb
    self.jumping = config.model.jumping
    self.skip_connection = config.model.skip_connection
    self.interpol = config.model.interpol

    assert self.aggregate_type in ['att', 'mean', 'add'], 'not implemented'

    #PROPAGATION LAYER
    if self.aggregate_type == 'att':
      self.GeoLayer = MYGATConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.degree_emb, self.skip_connection, self.interpol)
    else:
      self.GeoLayer = GeoLayer_msg(self.hidden_dim, self.degree_emb, self.aggregate_type, self.skip_connection, self.interpol)

    #OUTPUT FUNCTION
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])

    #JUMPING KNOWLEDGE
    self.output_JK = nn.Sequential(*[
      nn.Linear(self.hidden_dim * self.num_prop,  64),
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
      xx for xx in [self.output_func, self.output_JK] if xx is not None
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

  # def forward(self, J_msg, b, msg_node, degree, idx_msg_edge, target=None):
  def forward(self, data):
    """
      A: shape |V| X |V|
      J: shape |V| X |V|
      b: shape |V| X 1
      msg_node: shape |E| X 2
      msg_edge: shape |E| X |E|
      target: shape |V| X 2
    """
    J_msg, b, msg_node, idx_msg_edge, target = data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['y']
    num_node = b.shape[0]
    num_msg_node = msg_node.shape[1]
    # num_msg_edge = idx_msg_edge.shape[1]
    state = torch.zeros(num_msg_node, self.hidden_dim).to(b.device)

    # propagation
    hidden_state = []
    for tt in range(self.num_prop):
      if self.aggregate_type == 'att':
        # state = self.GeoLayer(state, msg_node.T, J_msg, b, degree, idx_msg_edge.T)
        state = self.GeoLayer(state, msg_node.T, J_msg, b, idx_msg_edge.T)
      else:
        # state = self.GeoLayer(msg_node.T, J_msg, b, state, degree, idx_msg_edge.T)
        state = self.GeoLayer(msg_node.T, J_msg, b, state, idx_msg_edge.T)

      if self.jumping:
        hidden_state.append(state)

    out_scatter_idx = msg_node.T[:, 1].view(-1, 1).expand(-1, self.hidden_dim)
    out_state_agg = torch.zeros(num_node, self.hidden_dim).to(b.device)
    out_state_agg = out_state_agg.scatter_add_(0, out_scatter_idx, state)
    y = self.output_func(torch.cat([out_state_agg, b, -b], dim=1))

    y = torch.log_softmax(y, dim=1)

    # if target is not None:
    #   if self.config.model.loss == 'KL-pq':
    #     # criterion(logQ, P) = KL(P||Q)
    #     loss = self.loss_func(y, target)
    #   elif self.config.model.loss == 'KL-qp':
    #     # criterion(logP, Q) = KL(Q||P)
    #     loss = self.loss_func(torch.log(target + EPS), torch.exp(y))
    #   elif self.config.model.loss == 'MSE':
    #     loss = self.loss_func(y, torch.log(target))
    #   else:
    #     raise ValueError("Non-supported loss function!")
    # else:
    #   loss = None

    return y

class GeoLayer_msg(MessagePassing):
  def __init__(self, hidden_dim, degree_emb, aggre_type, skip_connection=False, interpol=False):
    super(GeoLayer_msg, self).__init__(aggr=aggre_type)  # "Add" aggregation (Step 5).
    self.hidden_dim = hidden_dim
    self.degree_emb = degree_emb
    self.skip_connection = skip_connection
    self.interpol = interpol

    # message function
    c = 2
    self.msg_func = nn.Sequential(*[
      nn.Linear(c * self.hidden_dim + 16, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # degree embedding
    if self.degree_emb is True:
      self.degree_func = nn.Sequential(*[
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, self.hidden_dim)
      ])
    else:
      self.degree_func = None

    # update function
    self.update_func = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    self.update_down = nn.Sequential(*[
      nn.Linear(2*self.hidden_dim, self.hidden_dim),
      nn.ReLU()
    ])

    self.gating = nn.Sequential(*[
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.Sigmoid()
    ])

    self._init_param()

  def _init_param(self):
    mlp_modules = [
      xx for xx in [self.msg_func, self.degree_func, self.update_func, self.update_down, self.gating] if xx is not None
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

  # def forward(self, msg_node, J_msg, b, state_prev, degree, idx_msg_edge):
  def forward(self, msg_node, J_msg, b, state_prev, idx_msg_edge):
    edge_in = idx_msg_edge[:, 0]
    edge_out = idx_msg_edge[:, 1].contiguous()

    b_in = b[msg_node[:, 0]].view(-1, 1)  # shape |E| X 1
    b_out = b[msg_node[:, 1]].view(-1, 1)  # shape |E| X 1

    ff_in = torch.cat([b_in[edge_in], -b_in[edge_in], b_out[edge_in], -b_out[edge_in],
                       J_msg[edge_in], -J_msg[edge_in], -J_msg[edge_in], J_msg[edge_in]], dim=1)
    ff_out = torch.cat([b_in[edge_out], -b_in[edge_out], b_out[edge_out], -b_out[edge_out],
                        J_msg[edge_out], -J_msg[edge_out], -J_msg[edge_out], J_msg[edge_out]], dim=1)
    # ff = torch.cat([J_msg, b[edge_in], b[edge_out]], dim=1).float()

    state_in = state_prev[edge_in, :]  # shape |E| X D
    state_out = state_prev[edge_out, :]  # shape |E| X D
    # node_degree = degree[edge_out].float()
    # node_degree = self.degree_func(degree[edge_out].view(-1,1).float()) if self.degree_emb is True else 0
    #print(max(idx_msg_edge.cpu()[:,0]), max(idx_msg_edge.cpu()[:,1]), idx_msg_edge.cpu().size())
    return self.propagate(edge_index = idx_msg_edge.t(), state_prev=state_prev, J_msg=J_msg, state_in=state_in,
                          state_out=state_out,
                          ff_in=ff_in, ff_out=ff_out) #,node_degree=node_degree)

  def message(self, state_in, state_out, ff_in, ff_out):
    out = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1))
    return out

  def update(self, msg_agg, ff_out, state_prev):
    #print(msg_agg.cpu().size(), ff_out.cpu().size())
    # msg = self.msg_func(torch.cat([msg_agg, ff_out], dim=1))
    out = self.update_func(msg_agg, state_prev)
    # skip-connection
    if self.skip_connection:
      if self.interpol:
        # skip-connection(interpol)
        alpha_inter = self.gating(state_prev)
        out = alpha_inter * out + (1 - alpha_inter) * state_prev
      else:
        # skip-connection
        out = torch.cat([out, state_prev], dim=-1)
        out = self.update_down(out)
    return out


