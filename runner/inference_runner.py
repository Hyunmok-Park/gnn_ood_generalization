from __future__ import (division, print_function)

import collections
import copy
import os
import time

import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import networkx as nx

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from model import *
from dataset import *
from torch_geometric.transforms import ToSparseTensor
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.topology import NetworkTopology, get_msg_graph
#from torch_geometric.nn import DataParallel
from utils.myparallel import DataParallel
import shutil

logger = get_logger('exp_logger')
EPS = float(np.finfo(np.float32).eps)
__all__ = ['NeuralInferenceRunner', 'NeuralInferenceRunner_Meta', 'NeuralInferenceRunner_Meta2', 'NeuralInferenceRunner_Meta3', 'AlgorithmicInferenceRunner_bp', 'AlgorithmicInferenceRunner', 'NeuralInferenceRunner_MT']

class NeuralInferenceRunner(object):
  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.shuffle = config.train.shuffle
    self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.SSL = config.model.SSL
    self.train_pretext = config.model.train_pretext

  @property
  def train(self):
    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(self.model_conf.name)(self.config)

    # fix parameter(pretext)
    if self.SSL and not self.train_pretext:
      print("Fixing parameter")
      for name, param in model.named_parameters():
        if "output_func" not in name:
          param.requires_grad = False
        if "pretext_output_func" in name:
          param.requires_grad = True

    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
        params,
        lr=self.train_conf.lr,
        momentum=self.train_conf.momentum,
        weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
        optimizer = optim.Adam(
          params,
          lr=self.train_conf.lr,
          weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=self.train_conf.lr_decay_steps,
      gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # ===================== validation ============================ #
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        for data in tqdm(val_loader):
          #data = ToSparseTensor(remove_edge_index=False)(data)
          #if self.use_gpu:
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
              data['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
          #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['idx_msg_edge'] = data_to_gpu(
          #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['idx_msg_edge'])

          with torch.no_grad():
            # _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['degree'], data['idx_msg_edge'], target=data['y'])
            if self.parallel:
              _ = model(data)
              target = torch.cat([i.y for i in data]).to(_.device)
              loss = nn.KLDivLoss(reduction='batchmean')(_, target)
            else:
              if self.SSL:
                if not self.train_pretext:
                  _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['y'])
                else:
                  _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['degree'])
              elif self.model_conf.name == "TorchGNN_multitask":
                _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], [data['degree'],data['y']])
              else:
                _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['y'])

            val_loss += [float(loss.data.cpu().numpy())]

        val_loss = np.stack(val_loss).mean()
        results['val_loss'] += [val_loss]
        logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(val_loss, 0, epoch))
        self.writer.add_scalar('val_loss', val_loss, iter_count)

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag="best")

        logger.info("Current Best Validation Loss = {}".format(best_val_loss))

        # check early stop
        if early_stop.tick([val_loss]):
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag='last')
          self.writer.close()
          break
      # ====================== training ============================= #
      model.train()
      lr_scheduler.step()
      for data in train_loader:
        # 0. clears all gradients.
        optimizer.zero_grad()
        #data = ToSparseTensor(remove_edge_index=False)(data)
        # if self.use_gpu:
        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
        #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['idx_msg_edge'] = data_to_gpu(
        #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['idx_msg_edge'])

        # 1. forward pass
        # 2. compute loss
        # _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['degree'], data['idx_msg_edge'], target=data['y'])
        if self.parallel:
          _ = model(data)
          target = torch.cat([i.y for i in data]).to(_.device)
          loss = nn.KLDivLoss(reduction='batchmean')(_, target)
        else:
          if self.SSL:
            if not self.train_pretext:
              _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['y'])
            else:
              _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], data['degree'])
          elif self.model_conf.name == "TorchGNN_multitask":
            _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], [data['degree'], data['y']])
          else:
            _, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], target=data['y'])

        # 3. backward pass (accumulates gradients).
        loss.backward()
        # 4. performs a single update step.
        optimizer.step()

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1
      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

    print(np.array(results['hidden_state']).shape)
    results['best_val_loss'] += [best_val_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {}".format(best_val_loss))

    return best_val_loss

  def test(self):
    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)

    # create data loader
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle = False)

    # create models
    model = eval(self.model_conf.name)(self.config, test=True)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    state_hist = []
    for data in tqdm(test_loader):
      # data = ToSparseTensor(remove_edge_index=False)(data)
      # if self.use_gpu:
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
      #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['degree'], data['idx_msg_edge'] = data_to_gpu(
      #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['degree'], data['idx_msg_edge'])

      with torch.no_grad():
        # log_prob, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['degree'], data['idx_msg_edge'], target=data['y'])
        log_prob, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], target=data['y'])

        loss = 0 if loss < 0 else float(loss.data.cpu().numpy())
        test_loss += [loss]

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    test_loss = np.stack(test_loss).mean()
    logger.info("Avg. Test Loss = {} +- {}".format(test_loss, 0))

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    state_hist = np.array(state_hist)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss

class NeuralInferenceRunner_Meta(object):
  def __init__(self, config, config2):
    self.config = config
    self.dataset_conf = config.dataset

    self.config2 = config2
    self.dataset_conf2 = config2.dataset

    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.shuffle = config.train.shuffle
    self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.SSL = config.model.SSL
    self.train_pretext = config.model.train_pretext

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = self.train_conf.max_epoch
    self.temp_update = self.model_conf.temp

  @property
  def train(self):
    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv


    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    train_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)
    val_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(self.model_conf.name)(self.config)

    # fix parameter(pretext)
    if self.SSL and not self.train_pretext:
      print("Fixing parameter")
      for name, param in model.named_parameters():
        if "output_func" not in name:
          param.requires_grad = False
        if "pretext_output_func" in name:
          param.requires_grad = True

    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
        params,
        lr=self.train_conf.lr,
        momentum=self.train_conf.momentum,
        weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
        optimizer = optim.Adam(
          params,
          lr=self.train_conf.lr,
          weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=self.train_conf.lr_decay_steps,
      gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)

    temp = np.exp(self.initial_temp)

    for epoch in range(self.train_conf.max_epoch):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      # ===================== validation ============================ #
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
              data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
              data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

          ###############################
          # DO NOT SEARCH VAL STRUCTURE
          ###############################
          node_idx_inv1 = [0 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
          node_idx1 = [[_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)], []]
          node_idx_inv2 = [1 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
          node_idx2 = [[], [_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]]

          ########################
          # SEARCH VAL STRUCTURE
          ########################
          # node_idx1, node_idx_inv1 = node_idx_to_batch(data1['node_idx'], data1['node_idx_inv'])
          # node_idx2, node_idx_inv2 = node_idx_to_batch(data2['node_idx'], data2['node_idx_inv'])

          with torch.no_grad():
            _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
            _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

            # new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(data1)
            # new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(data2)
            #
            # new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
            # new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)
            #
            # _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
            #                               target=data1['y'], node_idx=new_node_idx1_, node_idx_inv=new_node_idx_inv1_)
            # _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
            #                               target=data2['y'], node_idx=new_node_idx2_, node_idx_inv=new_node_idx_inv2_)
            #
            # updated_node_idx1 = []
            # updated_node_idx_inv1 = []
            # for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1, new_loss_batch1):
            #
            #   if self.temp_update:
            #     prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
            #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            #   else:
            #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()
            #
            #   if accept:
            #     updated_node_idx1.append(new_node_idx1[idx])
            #     updated_node_idx_inv1.append(new_node_idx_inv1[idx])
            #     val_loss += [float(new_loss.data.cpu().numpy())]
            #   else:
            #     updated_node_idx1.append(data1['node_idx'][idx])
            #     updated_node_idx_inv1.append(data1['node_idx_inv'][idx])
            #     val_loss += [float(old_loss.data.cpu().numpy())]
            #
            # updated_node_idx2 = []
            # updated_node_idx_inv2 = []
            # for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
            #                                    new_loss_batch2):
            #
            #   if self.temp_update:
            #     prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
            #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            #   else:
            #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()
            #
            #   if accept:
            #     updated_node_idx2.append(new_node_idx1[idx])
            #     updated_node_idx_inv2.append(new_node_idx_inv1[idx])
            #     val_loss += [float(new_loss.data.cpu().numpy())]
            #   else:
            #     updated_node_idx2.append(data2['node_idx'][idx])
            #     updated_node_idx_inv2.append(data2['node_idx_inv'][idx])
            #     val_loss += [float(old_loss.data.cpu().numpy())]
            #
            # data1['node_idx'] = updated_node_idx1
            # data2['node_idx'] = updated_node_idx2
            # data1['node_idx_inv'] = updated_node_idx_inv1
            # data2['node_idx_inv'] = updated_node_idx_inv2

            # if loss1 > new_loss1:
            #   val_loss += [float(new_loss1.data.cpu().numpy())]
            # else:
            #   val_loss += [float(loss1.data.cpu().numpy())]
            #
            # if loss2 > new_loss2:
            #   val_loss += [float(new_loss2.data.cpu().numpy())]
            # else:
            #   val_loss += [float(loss2.data.cpu().numpy())]
            val_loss += [float(loss1.data.cpu().numpy())]
            val_loss += [float(loss2.data.cpu().numpy())]

        val_loss = np.stack(val_loss).mean()
        results['val_loss'] += [val_loss]
        logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(val_loss, 0, epoch))
        self.writer.add_scalar('val_loss', val_loss, iter_count)

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag="best")

        logger.info("Current Best Validation Loss = {}".format(best_val_loss))

        # check early stop
        if early_stop.tick([val_loss]):
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag='last')
          self.writer.close()
          break
      # ====================== training ============================= #
      model.train()
      lr_scheduler.step()
      for data1, data2 in zip(train_loader1, train_loader2):
        # 0. clears all gradients.
        loss = 0
        optimizer.zero_grad()
        # if self.use_gpu:
        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
          data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        # 1. forward pass
        # 2. compute loss

        ###############################
        # DO NOT SEARCH TRAIN STRUCTURE
        ###############################
        node_idx_inv1 = [0 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
        node_idx1 = [[_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)], []]
        node_idx_inv2 = [1 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
        node_idx2 = [[], [_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]]

        ########################
        # SEARCH TRAIN STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(data1['node_idx'], data1['node_idx_inv'])
        node_idx2, node_idx_inv2 = node_idx_to_batch(data2['node_idx'], data2['node_idx_inv'])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)
        #
        # new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(data1)
        # new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(data2)
        #
        # new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
        # new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)
        #
        # _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'],
        #                                       data1['idx_msg_edge'],
        #                                       target=data1['y'], node_idx=new_node_idx1_,
        #                                       node_idx_inv=new_node_idx_inv1_)
        # _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'],
        #                                       data2['idx_msg_edge'],
        #                                       target=data2['y'], node_idx=new_node_idx2_,
        #                                       node_idx_inv=new_node_idx_inv2_)
        #
        # updated_node_idx1 = []
        # updated_node_idx_inv1 = []
        # for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1,
        #                                    new_loss_batch1):
        #   if self.temp_update:
        #     prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
        #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
        #     upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
        #     self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(), new_loss.data.cpu().numpy(), upt_factor, accept, self.SA_running_factor, self.SA_running_acc_rate)
        #   else:
        #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()
        #
        #   if accept:
        #     updated_node_idx1.append(new_node_idx1[idx])
        #     updated_node_idx_inv1.append(new_node_idx_inv1[idx])
        #     loss += new_loss
        #
        #   else:
        #     updated_node_idx1.append(data1['node_idx'][idx])
        #     updated_node_idx_inv1.append(data1['node_idx_inv'][idx])
        #     loss += old_loss
        #
        # updated_node_idx2 = []
        # updated_node_idx_inv2 = []
        # for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
        #                                    new_loss_batch2):
        #
        #   if self.temp_update:
        #     prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
        #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
        #     upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
        #     self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
        #                                                                                  new_loss.data.cpu().numpy(),
        #                                                                                  upt_factor, accept,
        #                                                                                  self.SA_running_factor,
        #                                                                                  self.SA_running_acc_rate)
        #   else:
        #     accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()
        #
        #   if accept:
        #     updated_node_idx2.append(new_node_idx1[idx])
        #     updated_node_idx_inv2.append(new_node_idx_inv1[idx])
        #     loss += new_loss
        #   else:
        #     updated_node_idx2.append(data2['node_idx'][idx])
        #     updated_node_idx_inv2.append(data2['node_idx_inv'][idx])
        #     loss += old_loss
        #
        # data1['node_idx'] = updated_node_idx1
        # data2['node_idx'] = updated_node_idx2
        # data1['node_idx_inv'] = updated_node_idx_inv1
        # data2['node_idx_inv'] = updated_node_idx_inv2

        # loss = torch.min(loss1, new_loss1) + torch.min(loss2, new_loss2)
        loss = loss1 + loss2

        # 3. backward pass (accumulates gradients).
        loss.backward()
        # 4. performs a single update step.
        optimizer.step()

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1
      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

    val_structure1 = []
    train_structure1 = []
    val_structure2 = []
    train_structure2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      val_structure1.append([data1['edge_index'], data1['node_idx']])
      val_structure2.append([data2['edge_index'], data2['node_idx']])
    for data1, data2 in zip(train_loader1, train_loader2):
      train_structure1.append([data1['edge_index'], data1['node_idx']])
      train_structure2.append([data2['edge_index'], data2['node_idx']])

    with open(os.path.join(self.config.save_dir, 'val_structure1.p'), "wb") as f:
      pickle.dump(val_structure1, f)
      del val_structure1

    with open(os.path.join(self.config.save_dir, 'val_structure2.p'), "wb") as f:
      pickle.dump(val_structure2, f)
      del val_structure2

    with open(os.path.join(self.config.save_dir, 'train_structure1.p'), "wb") as f:
      pickle.dump(train_structure1, f)
      del train_structure1

    with open(os.path.join(self.config.save_dir, 'train_structure2.p'), "wb") as f:
      pickle.dump(train_structure2, f)
      del train_structure2


    print(np.array(results['hidden_state']).shape)
    results['best_val_loss'] += [best_val_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {}".format(best_val_loss))

    return best_val_loss

  def test(self):

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = 500

    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)
    tik = time.time()

    # create data loader
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False)

    # create models
    model = eval(self.model_conf.name)(self.config, test=True)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    state_hist = []
    structure_info = []

    temp = np.exp(self.initial_temp)

    for step in tqdm(range(500), desc="META TEST"):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * step / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      for data in tqdm(test_loader):
        loss = 0
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():
          # log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])
          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(data)
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          # old_node_idx, old_node_idx_inv = copy.deepcopy(data['node_idx']), copy.deepcopy(data['node_idx_inv'])
          # new_node_idx, new_node_idx_inv = propose_new_sturucture(data)
          # data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv

          # log_prob_new, loss_new, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
          #                       target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          # if loss > loss_new:
          #   data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv
          # else:
          #   data['node_idx'], data['node_idx_inv'] = old_node_idx, old_node_idx_inv



          updated_node_idx = []
          updated_node_idx_inv = []
          for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch,
                                             new_loss_batch):

            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
                                                                                           new_loss.data.cpu().numpy(),
                                                                                           upt_factor, accept,
                                                                                           self.SA_running_factor,
                                                                                           self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx.append(new_node_idx[idx])
              updated_node_idx_inv.append(new_node_idx_inv[idx])
              loss += new_loss
            else:
              updated_node_idx.append(data['node_idx'][idx])
              updated_node_idx_inv.append(data['node_idx_inv'][idx])
              loss += old_loss

          data['node_idx'] = updated_node_idx
          data['node_idx_inv'] = updated_node_idx_inv

        logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for data in tqdm(test_loader):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]
        structure_info.append(data['node_idx_inv'])

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    state_hist = np.array(state_hist)
    # structure_info = np.array(structure_info)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "sturucture_info.p")
    with open(file_name, 'wb') as f:
      pickle.dump(structure_info, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss

class NeuralInferenceRunner_Meta2(object):
  def __init__(self, config, config2):
    self.config = config
    self.dataset_conf = config.dataset

    self.config2 = config2
    self.dataset_conf2 = config2.dataset

    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.shuffle = config.train.shuffle
    self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.SSL = config.model.SSL
    self.train_pretext = config.model.train_pretext

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = self.train_conf.max_epoch
    self.temp_update = self.model_conf.temp

  @property
  def train(self):
    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv


    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    train_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)
    val_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(self.model_conf.name)(self.config)

    # fix parameter(pretext)
    if self.SSL and not self.train_pretext:
      print("Fixing parameter")
      for name, param in model.named_parameters():
        if "output_func" not in name:
          param.requires_grad = False
        if "pretext_output_func" in name:
          param.requires_grad = True

    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
        params,
        lr=self.train_conf.lr,
        momentum=self.train_conf.momentum,
        weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
        optimizer = optim.Adam(
          params,
          lr=self.train_conf.lr,
          weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=self.train_conf.lr_decay_steps,
      gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)

    temp = np.exp(self.initial_temp)

    for epoch in range(self.train_conf.max_epoch):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      # ===================== validation ============================ #
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
              data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
              data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

          ###############################
          # DO NOT SEARCH VAL STRUCTURE
          ###############################
          # node_idx_inv1 = [0 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
          # node_idx1 = [[_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)], []]
          # node_idx_inv2 = [1 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
          # node_idx2 = [[], [_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]]

          ########################
          # SEARCH VAL STRUCTURE
          ########################
          node_idx1, node_idx_inv1 = node_idx_to_batch(data1['node_idx'], data1['node_idx_inv'])
          node_idx2, node_idx_inv2 = node_idx_to_batch(data2['node_idx'], data2['node_idx_inv'])

          with torch.no_grad():
            _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
            _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

            new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(data1)
            new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(data2)

            new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
            new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)

            _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
                                          target=data1['y'], node_idx=new_node_idx1_, node_idx_inv=new_node_idx_inv1_)
            _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
                                          target=data2['y'], node_idx=new_node_idx2_, node_idx_inv=new_node_idx_inv2_)

            updated_node_idx1 = []
            updated_node_idx_inv1 = []
            for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1, new_loss_batch1):

              if self.temp_update:
                prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
                accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              else:
                accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

              if accept:
                updated_node_idx1.append(new_node_idx1[idx])
                updated_node_idx_inv1.append(new_node_idx_inv1[idx])
                val_loss += [float(new_loss.data.cpu().numpy())]
              else:
                updated_node_idx1.append(data1['node_idx'][idx])
                updated_node_idx_inv1.append(data1['node_idx_inv'][idx])
                val_loss += [float(old_loss.data.cpu().numpy())]

            updated_node_idx2 = []
            updated_node_idx_inv2 = []
            for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
                                               new_loss_batch2):

              if self.temp_update:
                prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
                accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              else:
                accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

              if accept:
                updated_node_idx2.append(new_node_idx1[idx])
                updated_node_idx_inv2.append(new_node_idx_inv1[idx])
                val_loss += [float(new_loss.data.cpu().numpy())]
              else:
                updated_node_idx2.append(data2['node_idx'][idx])
                updated_node_idx_inv2.append(data2['node_idx_inv'][idx])
                val_loss += [float(old_loss.data.cpu().numpy())]

            data1['node_idx'] = updated_node_idx1
            data2['node_idx'] = updated_node_idx2
            data1['node_idx_inv'] = updated_node_idx_inv1
            data2['node_idx_inv'] = updated_node_idx_inv2

            # if loss1 > new_loss1:
            #   val_loss += [float(new_loss1.data.cpu().numpy())]
            # else:
            #   val_loss += [float(loss1.data.cpu().numpy())]
            #
            # if loss2 > new_loss2:
            #   val_loss += [float(new_loss2.data.cpu().numpy())]
            # else:
            #   val_loss += [float(loss2.data.cpu().numpy())]
            # val_loss += [float(loss1.data.cpu().numpy())]
            # val_loss += [float(loss2.data.cpu().numpy())]

        val_loss = np.stack(val_loss).mean()
        results['val_loss'] += [val_loss]
        logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(val_loss, 0, epoch))
        self.writer.add_scalar('val_loss', val_loss, iter_count)

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag="best")

        logger.info("Current Best Validation Loss = {}".format(best_val_loss))

        # check early stop
        if early_stop.tick([val_loss]):
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag='last')
          self.writer.close()
          break
      # ====================== training ============================= #
      model.train()
      lr_scheduler.step()
      for data1, data2 in zip(train_loader1, train_loader2):
        # 0. clears all gradients.
        loss = 0
        optimizer.zero_grad()
        # if self.use_gpu:
        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
          data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        # 1. forward pass
        # 2. compute loss

        ###############################
        # DO NOT SEARCH TRAIN STRUCTURE
        ###############################
        # node_idx_inv1 = [0 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
        # node_idx1 = [[_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)], []]
        # node_idx_inv2 = [1 for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]
        # node_idx2 = [[], [_ for _ in range(self.dataset_conf.num_node * self.train_conf.batch_size)]]

        ########################
        # SEARCH TRAIN STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(data1['node_idx'], data1['node_idx_inv'])
        node_idx2, node_idx_inv2 = node_idx_to_batch(data2['node_idx'], data2['node_idx_inv'])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

        new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(data1)
        new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(data2)

        new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
        new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)

        _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'],
                                              data1['idx_msg_edge'],
                                              target=data1['y'], node_idx=new_node_idx1_,
                                              node_idx_inv=new_node_idx_inv1_)
        _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'],
                                              data2['idx_msg_edge'],
                                              target=data2['y'], node_idx=new_node_idx2_,
                                              node_idx_inv=new_node_idx_inv2_)

        updated_node_idx1 = []
        updated_node_idx_inv1 = []
        for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1,
                                           new_loss_batch1):
          if self.temp_update:
            prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
            self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(), new_loss.data.cpu().numpy(), upt_factor, accept, self.SA_running_factor, self.SA_running_acc_rate)
          else:
            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

          if accept:
            updated_node_idx1.append(new_node_idx1[idx])
            updated_node_idx_inv1.append(new_node_idx_inv1[idx])
            loss += new_loss

          else:
            updated_node_idx1.append(data1['node_idx'][idx])
            updated_node_idx_inv1.append(data1['node_idx_inv'][idx])
            loss += old_loss

        updated_node_idx2 = []
        updated_node_idx_inv2 = []
        for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
                                           new_loss_batch2):

          if self.temp_update:
            prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
            self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
                                                                                         new_loss.data.cpu().numpy(),
                                                                                         upt_factor, accept,
                                                                                         self.SA_running_factor,
                                                                                         self.SA_running_acc_rate)
          else:
            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

          if accept:
            updated_node_idx2.append(new_node_idx1[idx])
            updated_node_idx_inv2.append(new_node_idx_inv1[idx])
            loss += new_loss
          else:
            updated_node_idx2.append(data2['node_idx'][idx])
            updated_node_idx_inv2.append(data2['node_idx_inv'][idx])
            loss += old_loss

        data1['node_idx'] = updated_node_idx1
        data2['node_idx'] = updated_node_idx2
        data1['node_idx_inv'] = updated_node_idx_inv1
        data2['node_idx_inv'] = updated_node_idx_inv2

        # loss = torch.min(loss1, new_loss1) + torch.min(loss2, new_loss2)
        # loss = loss1 + loss2

        # 3. backward pass (accumulates gradients).
        loss.backward()
        # 4. performs a single update step.
        optimizer.step()

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1
      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

    val_structure1 = []
    train_structure1 = []
    val_structure2 = []
    train_structure2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      val_structure1.append([data1['edge_index'], data1['node_idx']])
      val_structure2.append([data2['edge_index'], data2['node_idx']])
    for data1, data2 in zip(train_loader1, train_loader2):
      train_structure1.append([data1['edge_index'], data1['node_idx']])
      train_structure2.append([data2['edge_index'], data2['node_idx']])

    with open(os.path.join(self.config.save_dir, 'val_structure1.p'), "wb") as f:
      pickle.dump(val_structure1, f)
      del val_structure1

    with open(os.path.join(self.config.save_dir, 'val_structure2.p'), "wb") as f:
      pickle.dump(val_structure2, f)
      del val_structure2

    with open(os.path.join(self.config.save_dir, 'train_structure1.p'), "wb") as f:
      pickle.dump(train_structure1, f)
      del train_structure1

    with open(os.path.join(self.config.save_dir, 'train_structure2.p'), "wb") as f:
      pickle.dump(train_structure2, f)
      del train_structure2


    print(np.array(results['hidden_state']).shape)
    results['best_val_loss'] += [best_val_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {}".format(best_val_loss))

    return best_val_loss

  def test(self):

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = 500

    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)
    tik = time.time()

    # create data loader
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False)

    # create models
    model = eval(self.model_conf.name)(self.config, test=True)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    state_hist = []
    structure_info = []

    temp = np.exp(self.initial_temp)

    for step in tqdm(range(500), desc="META TEST"):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * step / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      for data in tqdm(test_loader):
        loss = 0
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():
          # log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])
          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(data)
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          # old_node_idx, old_node_idx_inv = copy.deepcopy(data['node_idx']), copy.deepcopy(data['node_idx_inv'])
          # new_node_idx, new_node_idx_inv = propose_new_sturucture(data)
          # data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv

          # log_prob_new, loss_new, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
          #                       target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          # if loss > loss_new:
          #   data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv
          # else:
          #   data['node_idx'], data['node_idx_inv'] = old_node_idx, old_node_idx_inv



          updated_node_idx = []
          updated_node_idx_inv = []
          for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch,
                                             new_loss_batch):

            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
                                                                                           new_loss.data.cpu().numpy(),
                                                                                           upt_factor, accept,
                                                                                           self.SA_running_factor,
                                                                                           self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx.append(new_node_idx[idx])
              updated_node_idx_inv.append(new_node_idx_inv[idx])
              loss += new_loss
            else:
              updated_node_idx.append(data['node_idx'][idx])
              updated_node_idx_inv.append(data['node_idx_inv'][idx])
              loss += old_loss

          data['node_idx'] = updated_node_idx
          data['node_idx_inv'] = updated_node_idx_inv

        logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for data in tqdm(test_loader):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]
        structure_info.append(data['node_idx_inv'])

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    state_hist = np.array(state_hist)
    # structure_info = np.array(structure_info)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "sturucture_info.p")
    with open(file_name, 'wb') as f:
      pickle.dump(structure_info, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss

class NeuralInferenceRunner_Meta3(object):
  def __init__(self, config, config2):
    self.config = config
    self.dataset_conf = config.dataset

    self.config2 = config2
    self.dataset_conf2 = config2.dataset

    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.shuffle = config.train.shuffle
    self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.SSL = config.model.SSL
    self.train_pretext = config.model.train_pretext

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = self.train_conf.max_epoch
    self.temp_update = self.model_conf.temp

  @property
  def train(self):
    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv


    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    train_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)
    val_loader1, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader2, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(self.model_conf.name)(self.config)

    # fix parameter(pretext)
    if self.SSL and not self.train_pretext:
      print("Fixing parameter")
      for name, param in model.named_parameters():
        if "output_func" not in name:
          param.requires_grad = False
        if "pretext_output_func" in name:
          param.requires_grad = True

    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
        params,
        lr=self.train_conf.lr,
        momentum=self.train_conf.momentum,
        weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
        optimizer = optim.Adam(
          params,
          lr=self.train_conf.lr,
          weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=self.train_conf.lr_decay_steps,
      gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)
    temp = np.exp(self.initial_temp)

    for epoch in range(self.train_conf.max_epoch):
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      model.train()
      lr_scheduler.step()
      val_loss_list = []
      # ===================== validation ============================ #
      for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
        # 0. clears all gradients.
        optimizer.zero_grad()
        val_loss = 0

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

        ########################
        # SEARCH VAL STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(data1['node_idx'], data1['node_idx_inv'])
        node_idx2, node_idx_inv2 = node_idx_to_batch(data2['node_idx'], data2['node_idx_inv'])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

        for idx, old_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1):
          val_loss+=old_loss
        for idx, old_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2):
          val_loss+=old_loss

        val_loss.backward()
        optimizer.step()
        val_loss_list.append(val_loss.data.cpu().numpy())

      mean_loss = np.mean(val_loss_list)
      logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(mean_loss, 0, epoch))
      self.writer.add_scalar('val_loss', mean_loss, iter_count)
      results['val_loss']+=[mean_loss]

      # save best model
      if mean_loss < best_val_loss:
        best_val_loss = mean_loss
        snapshot(
          model.module if self.use_gpu else model,
          optimizer,
          self.config,
          epoch + 1,
          tag="best")

      logger.info("Current Best Validation Loss = {}".format(best_val_loss))

      # ====================== training ============================= #
      model.eval()
      for ii1, [data1, data2] in enumerate(zip(train_loader1, train_loader2)):
        loss = 0
        ########################
        # SEARCH TRAIN STRUCTURE
        ########################
        with torch.no_grad():
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

          ##############################
          # LOAD STRUCTURE FROM VAL DATA
          ##############################

          for ii2, [data1_val, data2_val] in enumerate(zip(val_loader1, val_loader2)):
            if ii2 == ii1 % 20:
              node_idx1, node_idx_inv1 = node_idx_to_batch(data1_val['node_idx'], data1_val['node_idx_inv'])
              node_idx2, node_idx_inv2 = node_idx_to_batch(data2_val['node_idx'], data2_val['node_idx_inv'])

              _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
              _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

              new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(data1_val)
              new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(data2_val)

              new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
              new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)

              _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'],
                                                    data1['idx_msg_edge'],
                                                    target=data1['y'], node_idx=new_node_idx1_,
                                                    node_idx_inv=new_node_idx_inv1_)
              _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'],
                                                    data2['idx_msg_edge'],
                                                    target=data2['y'], node_idx=new_node_idx2_,
                                                    node_idx_inv=new_node_idx_inv2_)

              updated_node_idx1 = []
              updated_node_idx_inv1 = []
              for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1,
                                                 new_loss_batch1):
                if self.temp_update:
                  prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
                  accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
                  upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
                  self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(), new_loss.data.cpu().numpy(), upt_factor, accept, self.SA_running_factor, self.SA_running_acc_rate)
                else:
                  accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

                if accept:
                  updated_node_idx1.append(new_node_idx1[idx])
                  updated_node_idx_inv1.append(new_node_idx_inv1[idx])
                  loss += new_loss

                else:
                  updated_node_idx1.append(data1_val['node_idx'][idx])
                  updated_node_idx_inv1.append(data1_val['node_idx_inv'][idx])
                  loss += old_loss

              updated_node_idx2 = []
              updated_node_idx_inv2 = []
              for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
                                                 new_loss_batch2):

                if self.temp_update:
                  prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
                  accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
                  upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
                  self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
                                                                                               new_loss.data.cpu().numpy(),
                                                                                               upt_factor, accept,
                                                                                               self.SA_running_factor,
                                                                                               self.SA_running_acc_rate)
                else:
                  accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

                if accept:
                  updated_node_idx2.append(new_node_idx1[idx])
                  updated_node_idx_inv2.append(new_node_idx_inv1[idx])
                  loss += new_loss
                else:
                  updated_node_idx2.append(data2_val['node_idx'][idx])
                  updated_node_idx_inv2.append(data2_val['node_idx_inv'][idx])
                  loss += old_loss

              data1_val['node_idx'] = updated_node_idx1
              data2_val['node_idx'] = updated_node_idx2
              data1_val['node_idx_inv'] = updated_node_idx_inv1
              data2_val['node_idx_inv'] = updated_node_idx_inv2
            else:
              continue

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

    val_structure1 = []
    train_structure1 = []
    val_structure2 = []
    train_structure2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      val_structure1.append([data1['edge_index'], data1['node_idx']])
      val_structure2.append([data2['edge_index'], data2['node_idx']])
    for data1, data2 in zip(train_loader1, train_loader2):
      train_structure1.append([data1['edge_index'], data1['node_idx']])
      train_structure2.append([data2['edge_index'], data2['node_idx']])

    with open(os.path.join(self.config.save_dir, 'val_structure1.p'), "wb") as f:
      pickle.dump(val_structure1, f)
      del val_structure1

    with open(os.path.join(self.config.save_dir, 'val_structure2.p'), "wb") as f:
      pickle.dump(val_structure2, f)
      del val_structure2

    with open(os.path.join(self.config.save_dir, 'train_structure1.p'), "wb") as f:
      pickle.dump(train_structure1, f)
      del train_structure1

    with open(os.path.join(self.config.save_dir, 'train_structure2.p'), "wb") as f:
      pickle.dump(train_structure2, f)
      del train_structure2


    print(np.array(results['hidden_state']).shape)
    results['best_val_loss'] += [best_val_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {}".format(best_val_loss))

    return best_val_loss

  def test(self):

    self.initial_temp = 0
    self.min_temp = -100
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = 500

    def update_frac_worse_accepts(old_loss, new_loss, upt_factor, accept, SA_running_factor, SA_running_acc_rate):
      if old_loss < new_loss:
        y = upt_factor if accept else 0
        SA_running_factor = ((1 - upt_factor) * SA_running_factor + upt_factor)
        SA_running_acc_rate = ((1 - upt_factor) * SA_running_acc_rate + y)
      return SA_running_factor, SA_running_acc_rate

    def propose_new_sturucture_batch(data):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        if change_node:
          idx = -1
          while idx == -1 or node_idx_inv[idx] >= 2:
            idx = np.random.randint(len(node_idx_inv))
          # Remove from old
          old_module = node_idx_inv[idx]
          pos_in_old = node_idx[old_module].index(idx)
          del node_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          node_idx_inv[idx] = new_module
          node_idx[new_module].append(idx)
        return node_idx, node_idx_inv

      new_node_idx = []
      new_node_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(data['node_idx'], data['node_idx_inv']):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      return new_node_idx, new_node_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)
    tik = time.time()

    # create data loader
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False)

    # create models
    model = eval(self.model_conf.name)(self.config, test=True)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    state_hist = []
    structure_info = []

    temp = np.exp(self.initial_temp)

    for step in tqdm(range(500), desc="META TEST"):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * step / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      for data in tqdm(test_loader):
        loss = 0
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():
          # log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])
          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(data)
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          # old_node_idx, old_node_idx_inv = copy.deepcopy(data['node_idx']), copy.deepcopy(data['node_idx_inv'])
          # new_node_idx, new_node_idx_inv = propose_new_sturucture(data)
          # data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv

          # log_prob_new, loss_new, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
          #                       target=data['y'], node_idx=data['node_idx'][0], node_idx_inv=data['node_idx_inv'][0])

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          # if loss > loss_new:
          #   data['node_idx'], data['node_idx_inv'] = new_node_idx, new_node_idx_inv
          # else:
          #   data['node_idx'], data['node_idx_inv'] = old_node_idx, old_node_idx_inv



          updated_node_idx = []
          updated_node_idx_inv = []
          for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch,
                                             new_loss_batch):

            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(old_loss.data.cpu().numpy(),
                                                                                           new_loss.data.cpu().numpy(),
                                                                                           upt_factor, accept,
                                                                                           self.SA_running_factor,
                                                                                           self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx.append(new_node_idx[idx])
              updated_node_idx_inv.append(new_node_idx_inv[idx])
              loss += new_loss
            else:
              updated_node_idx.append(data['node_idx'][idx])
              updated_node_idx_inv.append(data['node_idx_inv'][idx])
              loss += old_loss

          data['node_idx'] = updated_node_idx
          data['node_idx_inv'] = updated_node_idx_inv

        logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for data in tqdm(test_loader):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(data['node_idx'], data['node_idx_inv'])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]
        structure_info.append(data['node_idx_inv'])

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    state_hist = np.array(state_hist)
    # structure_info = np.array(structure_info)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "sturucture_info.p")
    with open(file_name, 'wb') as f:
      pickle.dump(structure_info, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss

class AlgorithmicInferenceRunner_bp(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.save_dir = config.save_dir
    self.num_graphs = 20

  def test(self):
    # create data loader
    # test_loader = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split)
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(model)

    loss_dict = []
    marginal_gt_dict = []
    marginal_pred_dict = []
    prob_step = []

    # model.eval()
    for idx, data in tqdm(enumerate(test_loader)):
      # if self.use_gpu:
      #   data['edge_attr'], data['x'], data['edge_index'], data['y'], data['edge_index'] = data_to_gpu(
      #     data['edge_attr'], data['x'], data['edge_index'], data['y'], data['edge_index'])
      # data['edge_attr'], data['x'], data['edge_index'], data['y'], data['edge_index'] = data['edge_attr'].to(device), data['x'].to(device), data['edge_index'].to(device), data['y'].to(device), data['edge_index'].to(device)

      with torch.no_grad():
        msg_node, msg_adj = get_msg_graph(nx.from_numpy_matrix(data['J'].numpy()))
        prob_pred, prob_step_ = BeliefPropagation(data['J'], data['x'], msg_node, msg_adj).inference(max_iter=15, damping=0.0,
                                                                                        observe_mask=None)

        marginal_gt_dict += [data['y'].data.cpu().numpy()]
        marginal_pred_dict += [prob_pred]
        prob_step += [prob_step_]

    pred_pts = np.concatenate(marginal_pred_dict, axis=0)
    gt_pts = np.concatenate(marginal_gt_dict, axis=0)
    prob_step = np.concatenate(prob_step, axis =0)

    file_name = os.path.join(self.save_dir, 'gt_pts.csv')
    np.savetxt(file_name, gt_pts, delimiter='\t')
    file_name = os.path.join(self.save_dir, 'pred_pts.csv')
    np.savetxt(file_name, pred_pts, delimiter='\t')
    file_name = os.path.join(self.save_dir, 'prob_steps.csv')
    np.savetxt(file_name, prob_step, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    return loss_dict

class AlgorithmicInferenceRunner(object):

  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.save_dir = config.save_dir
    self.num_graphs = 20

  def test(self):
    # create data loader
    test_loader, _ = eval(self.dataset_conf.loader_name)(self.config, split='test')

    # create models
    model = eval(self.model_conf.name)(self.config)
    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    loss_dict = []
    marginal_gt_dict = []
    marginal_pred_dict = []
    prob_step = []

    model.eval()
    for idx, data in tqdm(enumerate(test_loader)):
      if self.use_gpu:
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['msg_adj'] = data_to_gpu(
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['msg_adj'])

      with torch.no_grad():
        prob_pred, loss, prob_step_ = model(data['J'], data['x'], data['edge_index'], data['msg_adj'], target=data['y'])
        loss = 0 if loss < 0 else loss.data.cpu().numpy()

        loss_dict += [loss]
        marginal_gt_dict += [data['y'].data.cpu().numpy()]
        marginal_pred_dict += [prob_pred.data.cpu().numpy()]
        prob_step += [prob_step_]

    pred_pts = np.concatenate(marginal_pred_dict, axis=0)
    gt_pts = np.concatenate(marginal_gt_dict, axis=0)
    prob_step = np.concatenate(prob_step, axis =0)

    file_name = os.path.join(self.save_dir, 'gt_pts.csv')
    np.savetxt(file_name, gt_pts, delimiter='\t')
    file_name = os.path.join(self.save_dir, 'pred_pts.csv')
    np.savetxt(file_name, pred_pts, delimiter='\t')
    file_name = os.path.join(self.save_dir, 'prob_steps.csv')
    np.savetxt(file_name, prob_step, delimiter='\t')

    return loss_dict

class NeuralInferenceRunner_MT(object):
  def __init__(self, config):
    self.config = config
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.writer = SummaryWriter(config.save_dir)
    self.shuffle = config.train.shuffle
    self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
    self.master_node = config.model.master_node if config.model.master_node is not None else False
    self.train_pretext = config.model.train_pretext

  @property
  def train(self):
    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader_main, _ = eval(self.dataset_conf.loader_name)(self.config, split='train_main', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader_main, _ = eval(self.dataset_conf.loader_name)(self.config, split='val_main', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    train_loader_ssl, _ = eval(self.dataset_conf.loader_name)(self.config, split='train_ssl', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader_ssl, _ = eval(self.dataset_conf.loader_name)(self.config, split='val_ssl', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(self.model_conf.name)(self.config)

    if self.use_gpu:
      if self.parallel:
        print("Using GPU dataparallel")
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        model = DataParallel(model)
      else:
        print("Using single GPU")
        model = nn.DataParallel(model, device_ids=self.gpus)
        # model = DataParallel(model)
    model.to(device)
    print(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
        params,
        lr=self.train_conf.lr,
        momentum=self.train_conf.momentum,
        weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
        optimizer = optim.Adam(
          params,
          lr=self.train_conf.lr,
          weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optsimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=self.train_conf.lr_decay_steps,
      gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    if self.train_conf.is_resume:
      load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # ===================== validation ============================ #
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        for data_main, data_ssl in tqdm(zip(val_loader_main, val_loader_ssl)):
          with torch.no_grad():
            data_main['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
            data_ssl['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
            _, loss = model(data_main['edge_attr'], data_main['x'], data_main['edge_index'], data_main['idx_msg_edge'], 0, [data_main['y'],data_main['degree']])
            _, loss2 = model(data_ssl['edge_attr'], data_ssl['x'], data_ssl['edge_index'], data_ssl['idx_msg_edge'], 1, [data_ssl['y'],data_ssl['degree']])
            results['val_loss_seperate'] += [loss, loss2]
            loss = 0.5*loss + 0.5*loss2
            val_loss += [float(loss.data.cpu().numpy())]

        val_loss = np.stack(val_loss).mean()
        results['val_loss'] += [val_loss]
        logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(val_loss, 0, epoch))
        self.writer.add_scalar('val_loss', val_loss, iter_count)

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag="best")

        logger.info("Current Best Validation Loss = {}".format(best_val_loss))

        # check early stop
        if early_stop.tick([val_loss]):
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag='last')
          self.writer.close()
          break
      # ====================== training ============================= #
      model.train()
      lr_scheduler.step()
      for data_main, data_ssl in zip(train_loader_main, train_loader_ssl):
        # 0. clears all gradients.
        optimizer.zero_grad()

        # 1. forward pass
        # 2. compute loss
        data_main['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
        data_ssl['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
        _, loss = model(data_main['edge_attr'], data_main['x'], data_main['edge_index'], data_main['idx_msg_edge'], 0, [data_main['y'], data_main['degree']])
        _, loss2 = model(data_ssl['edge_attr'], data_ssl['x'], data_ssl['edge_index'], data_ssl['idx_msg_edge'], 1, [data_ssl['y'], data_ssl['degree']])
        results['train_loss_seperate'] += [loss, loss2]
        loss = 0.5*loss + 0.5*loss2

        # 3. backward pass (accumulates gradients).
        loss.backward()
        # 4. performs a single update step.
        optimizer.step()

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1
      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

      pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))

    print(np.array(results['hidden_state']).shape)
    results['best_val_loss'] += [best_val_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {}".format(best_val_loss))

    return best_val_loss

  def test(self):
    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)

    # create data loader
    test_loader, name_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle = False)

    # create models
    model = eval(self.model_conf.name)(self.config, test=True)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

    # create models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.use_gpu:
      print("Using single GPU")
      model = nn.DataParallel(model, device_ids=self.gpus)
      # model = DataParallel(model)
    model.to(device)
    print(model)

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    state_hist = []
    for data in tqdm(test_loader):
      data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
      with torch.no_grad():
        # log_prob, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['degree'], data['idx_msg_edge'], target=data['y'])

        log_prob, loss = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'], 0, [data['y'], data['degree']])
        loss = 0 if loss < 0 else float(loss.data.cpu().numpy())
        test_loss += [loss]

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    test_loss = np.stack(test_loss).mean()
    logger.info("Avg. Test Loss = {} +- {}".format(test_loss, 0))

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    state_hist = np.array(state_hist)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss