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
__all__ = ['NeuralInferenceRunner', 'NeuralInferenceRunner_Meta', 'NeuralInferenceRunner_Meta2', 'NeuralInferenceRunner_Meta3', 'NeuralInferenceRunner_Meta4','NeuralInferenceRunner_Meta3_zero', 'NeuralInferenceRunner_Meta4_zero', 'Modular_meta_learning', 'Modular_meta_learning_edge','AlgorithmicInferenceRunner_bp', 'AlgorithmicInferenceRunner', 'NeuralInferenceRunner_MT']

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
    train_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)
    val_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node)

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
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle = False)

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

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "state_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(state_hist, f)

    return test_loss

#DO NOT SEARCH TRAIN/VAL STRUCTURE
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

    return test_loss #

#SEARCH TRAIN/VAL STRUCTURE
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

#TRAIN NODE MODULE FROM VAL
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

    self.initial_temp = self.train_conf.init_temp
    self.min_temp = 0.00001
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

    def propose_new_sturucture_batch(node_idx, node_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        # change_node = False
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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
    train_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    train_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)

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

    node_module_hist = []

    node_idx_list1 = []
    node_idx_inv_list1 = []
    node_idx_list2 = []
    node_idx_inv_list2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      node_idx_list1.append(data1['node_idx'])
      node_idx_list2.append(data2['node_idx'])

      node_idx_inv_list1.append(data1['node_idx_inv'])
      node_idx_inv_list2.append(data2['node_idx_inv'])


    for epoch in range(self.train_conf.max_epoch):
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      model.eval()
      for idx_main, [data1, data2] in enumerate(zip(train_loader1, train_loader2)):
        loss = []
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
          node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
          node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

          _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
                                        target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
          _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
                                        target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

          new_node_idx1, new_node_idx_inv1 = propose_new_sturucture_batch(node_idx_list1[idx_main],
                                                                          node_idx_inv_list1[idx_main])
          new_node_idx2, new_node_idx_inv2 = propose_new_sturucture_batch(node_idx_list2[idx_main],
                                                                          node_idx_inv_list2[idx_main])

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
          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch1))], loss_batch1,
                                             new_loss_batch1):
            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(
                old_loss.data.cpu().numpy(), new_loss.data.cpu().numpy(), upt_factor, accept,
                self.SA_running_factor, self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx1.append(new_node_idx1[idx])
              updated_node_idx_inv1.append(new_node_idx_inv1[idx])
              loss += [float(new_loss.data.cpu().numpy())]

            else:
              updated_node_idx1.append(node_idx_list1[idx_main][idx])
              updated_node_idx_inv1.append(node_idx_inv_list1[idx_main][idx])
              loss += [float(old_loss.data.cpu().numpy())]

          updated_node_idx2 = []
          updated_node_idx_inv2 = []
          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch2))], loss_batch2,
                                             new_loss_batch2):

            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(
                old_loss.data.cpu().numpy(),
                new_loss.data.cpu().numpy(),
                upt_factor, accept,
                self.SA_running_factor,
                self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx2.append(new_node_idx2[idx])
              updated_node_idx_inv2.append(new_node_idx_inv2[idx])
              loss += [float(new_loss.data.cpu().numpy())]
            else:
              updated_node_idx2.append(node_idx_list2[idx_main][idx])
              updated_node_idx_inv2.append(node_idx_inv_list2[idx_main][idx])
              loss += [float(old_loss.data.cpu().numpy())]

          node_idx_list1[idx_main] = updated_node_idx1
          node_idx_inv_list1[idx_main] = updated_node_idx_inv1
          node_idx_list2[idx_main] = updated_node_idx2
          node_idx_inv_list2[idx_main] = updated_node_idx_inv2

        train_loss = np.stack(loss).mean()
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info(
            "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)


      model.train()
      lr_scheduler.step()
      val_loss_list = []
      # ===================== validation ============================ #
      for idx_main, [data1, data2] in tqdm(enumerate(zip(val_loader1, val_loader2))):
        # 0. clears all gradients.
        optimizer.zero_grad()

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

        ########################
        # SEARCH VAL STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
        node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

        node_module_hist.append([node_idx_inv1, node_idx_inv2])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

        val_loss = (loss1 + loss2) / 2

        val_loss.backward()
        optimizer.step()
        val_loss_list += [float(val_loss.data.cpu().numpy())]

      mean_loss = np.stack(val_loss_list).mean()
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

      snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        epoch + 1,
        tag="final")

      logger.info("Current Best Validation Loss = {}".format(best_val_loss))

    with open(os.path.join(self.config.save_dir, 'node_module_hist.p'), "wb") as f:
      pickle.dump(node_module_hist, f)
      del node_module_hist

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

    def propose_new_sturucture_batch(node_idx, node_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        # change_node = False
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, random_init=self.model_conf.random_init)

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
    pred_pts = []
    gt_pts = []
    node_module_hist = []
    loss_list = []

    temp = np.exp(self.initial_temp)

    node_idx_list = []
    node_idx_inv_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])

    for step in tqdm(range(self.config.test.optim_step), desc="META TEST"):
      test_loss = []
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * step / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      node_hist = copy.deepcopy(node_idx_inv_list)
      node_module_hist.append(node_hist)


      for idx_main, data in tqdm(enumerate(test_loader)):
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():
          loss_ = []
          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)


          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          updated_node_idx = []
          updated_node_idx_inv = []

          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch))], loss_batch,
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
              loss_ += [float(new_loss.data.cpu().numpy())]
            else:
              updated_node_idx.append(node_idx_list[idx_main][idx])
              updated_node_idx_inv.append(node_idx_inv_list[idx_main][idx])
              loss_ += [float(old_loss.data.cpu().numpy())]

          node_idx_list[idx_main] = updated_node_idx
          node_idx_inv_list[idx_main] = updated_node_idx_inv
          test_loss.append(np.stack(loss_).mean())

      mean_loss = np.mean(test_loss)
      loss_list.append(mean_loss)
      logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, mean_loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)

    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')

    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "node_module_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(node_module_hist, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return test_loss

#TRAIN NODE,EDGE MODULE FROM VAL
class NeuralInferenceRunner_Meta4(object):
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

    def propose_new_sturucture_batch(node_idx, node_idx_inv, edge_idx, edge_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        # change_node = False
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

      def propose_new_sturucture_edge(edge_idx, edge_idx_inv):
        change_edge = (np.random.rand() > 0.5)
        # change_edge = False
        if change_edge:
          idx = -1
          while idx == -1 or edge_idx_inv[idx] >= 2:
            idx = np.random.randint(len(edge_idx_inv))
          # Remove from old
          old_module = edge_idx_inv[idx]
          pos_in_old = edge_idx[old_module].index(idx)
          del edge_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          edge_idx_inv[idx] = new_module
          edge_idx[new_module].append(idx)
        return edge_idx, edge_idx_inv

      new_node_idx = []
      new_node_idx_inv = []
      new_edge_idx = []
      new_edge_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      for edge_idx_batch, edge_idx_inv_batch in zip(edge_idx, edge_idx_inv):
        new_edge_idx_batch, new_edge_idx_inv_batch = propose_new_sturucture_edge(edge_idx_batch, edge_idx_inv_batch)
        new_edge_idx.append(new_edge_idx_batch)
        new_edge_idx_inv.append(new_edge_idx_inv_batch)

      return new_node_idx, new_node_idx_inv, new_edge_idx, new_edge_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    def edge_idx_to_batch(edge_idx, edge_idx_inv):
      batched_edge_idx = [[], []]
      num_edge_before = 0
      for idx, idx_inv in zip(edge_idx, edge_idx_inv):
        batched_edge_idx[0] += [__ + num_edge_before for __ in idx[0]]
        batched_edge_idx[1] += [__ + num_edge_before for __ in idx[1]]

        num_edge_before += len(idx_inv)

      batched_edge_idx_inv = []
      for _ in edge_idx_inv:
        batched_edge_idx_inv += _

      return batched_edge_idx, batched_edge_idx_inv


    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    train_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)

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

    node_module_hist = []
    edge_module_hist = []

    node_idx_list1 = []
    node_idx_inv_list1 = []
    node_idx_list2 = []
    node_idx_inv_list2 = []

    edge_idx_list1 = []
    edge_idx_inv_list1 = []
    edge_idx_list2 = []
    edge_idx_inv_list2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      node_idx_list1.append(data1['node_idx'])
      node_idx_list2.append(data2['node_idx'])

      node_idx_inv_list1.append(data1['node_idx_inv'])
      node_idx_inv_list2.append(data2['node_idx_inv'])

      edge_idx_list1.append(data1['edge_idx'])
      edge_idx_list2.append(data2['edge_idx'])

      edge_idx_inv_list1.append(data1['edge_idx_inv'])
      edge_idx_inv_list2.append(data2['edge_idx_inv'])

    for epoch in range(self.train_conf.max_epoch):
      epoch_time = time.time()
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      # ====================== training ============================= #
      model.eval()
      for idx_main, [data1, data2] in enumerate(zip(train_loader1, train_loader2)):
        loss = []
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
          node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
          node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

          edge_idx1, edge_idx_inv1 = edge_idx_to_batch(edge_idx_list1[idx_main], edge_idx_inv_list1[idx_main])
          edge_idx2, edge_idx_inv2 = edge_idx_to_batch(edge_idx_list2[idx_main], edge_idx_inv_list2[idx_main])

          _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
                                        target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1,
                                        edge_idx=edge_idx1, edge_idx_inv=edge_idx_inv1)
          _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
                                        target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2,
                                        edge_idx=edge_idx2, edge_idx_inv=edge_idx_inv2)

          ######################
          # PROPOSE NEW STRUCTURE
          ######################
          new_node_idx1, new_node_idx_inv1, new_edge_idx1, new_edge_idx_inv1 = propose_new_sturucture_batch(
            node_idx_list1[idx_main], node_idx_inv_list1[idx_main], edge_idx_list1[idx_main],
            edge_idx_inv_list1[idx_main])
          new_node_idx2, new_node_idx_inv2, new_edge_idx2, new_edge_idx_inv2 = propose_new_sturucture_batch(
            node_idx_list2[idx_main], node_idx_inv_list2[idx_main], edge_idx_list2[idx_main],
            edge_idx_inv_list2[idx_main])

          new_node_idx1_, new_node_idx_inv1_ = node_idx_to_batch(new_node_idx1, new_node_idx_inv1)
          new_node_idx2_, new_node_idx_inv2_ = node_idx_to_batch(new_node_idx2, new_node_idx_inv2)

          new_edge_idx1_, new_edge_idx_inv1_ = edge_idx_to_batch(new_edge_idx1, new_edge_idx_inv1)
          new_edge_idx2_, new_edge_idx_inv2_ = edge_idx_to_batch(new_edge_idx2, new_edge_idx_inv2)

          _, new_loss1, new_loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'],
                                                data1['idx_msg_edge'],
                                                target=data1['y'], node_idx=new_node_idx1_,
                                                node_idx_inv=new_node_idx_inv1_, edge_idx=new_edge_idx1_,
                                                edge_idx_inv=new_edge_idx_inv1_)
          _, new_loss2, new_loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'],
                                                data2['idx_msg_edge'],
                                                target=data2['y'], node_idx=new_node_idx2_,
                                                node_idx_inv=new_node_idx_inv2_, edge_idx=new_edge_idx2_,
                                                edge_idx_inv=new_edge_idx_inv2_)

          updated_node_idx1 = []
          updated_node_idx_inv1 = []
          updated_edge_idx1 = []
          updated_edge_idx_inv1 = []
          for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch1,
                                             new_loss_batch1):
            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(
                old_loss.data.cpu().numpy(), new_loss.data.cpu().numpy(), upt_factor, accept,
                self.SA_running_factor, self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx1.append(new_node_idx1[idx])
              updated_node_idx_inv1.append(new_node_idx_inv1[idx])
              updated_edge_idx1.append(new_edge_idx1[idx])
              updated_edge_idx_inv1.append(new_edge_idx_inv1[idx])
              loss += [float(new_loss.data.cpu().numpy())]

            else:
              updated_node_idx1.append(node_idx_list1[idx_main][idx])
              updated_node_idx_inv1.append(node_idx_inv_list1[idx_main][idx])
              updated_edge_idx1.append(edge_idx_list1[idx_main][idx])
              updated_edge_idx_inv1.append(edge_idx_inv_list1[idx_main][idx])
              loss += [float(old_loss.data.cpu().numpy())]

          updated_node_idx2 = []
          updated_node_idx_inv2 = []
          updated_edge_idx2 = []
          updated_edge_idx_inv2 = []
          for idx, old_loss, new_loss in zip([_ for _ in range(self.train_conf.batch_size)], loss_batch2,
                                             new_loss_batch2):

            if self.temp_update:
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              self.SA_running_factor, self.SA_running_acc_rate = update_frac_worse_accepts(
                old_loss.data.cpu().numpy(),
                new_loss.data.cpu().numpy(),
                upt_factor, accept,
                self.SA_running_factor,
                self.SA_running_acc_rate)
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              updated_node_idx2.append(new_node_idx2[idx])
              updated_node_idx_inv2.append(new_node_idx_inv2[idx])
              updated_edge_idx2.append(new_edge_idx2[idx])
              updated_edge_idx_inv2.append(new_edge_idx_inv2[idx])
              loss += [float(new_loss.data.cpu().numpy())]
            else:
              updated_node_idx2.append(node_idx_list2[idx_main][idx])
              updated_node_idx_inv2.append(node_idx_inv_list2[idx_main][idx])
              updated_edge_idx2.append(edge_idx_list2[idx_main][idx])
              updated_edge_idx_inv2.append(edge_idx_inv_list2[idx_main][idx])
              loss += [float(old_loss.data.cpu().numpy())]

          node_idx_list1[idx_main] = updated_node_idx1
          node_idx_list2[idx_main] = updated_node_idx2
          node_idx_inv_list1[idx_main] = updated_node_idx_inv1
          node_idx_inv_list2[idx_main] = updated_node_idx_inv2

          edge_idx_list1[idx_main] = updated_edge_idx1
          edge_idx_list2[idx_main] = updated_edge_idx2
          edge_idx_inv_list1[idx_main] = updated_edge_idx_inv1
          edge_idx_inv_list2[idx_main] = updated_edge_idx_inv2

        train_loss = np.stack(loss).mean()
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info(
            "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

      # save best model
      if mean_loss < best_train_loss:
        best_val_loss = mean_loss
        snapshot(
          model.module if self.use_gpu else model,
          optimizer,
          self.config,
          epoch + 1,
          tag="best")

        snapshot(
          model.module if self.use_gpu else model,
          optimizer,
          self.config,
          epoch + 1,
          tag="final")

      model.train()
      lr_scheduler.step()
      val_loss_list = []
      # ===================== validation ============================ #
      for idx_main, [data1, data2] in tqdm(enumerate(zip(val_loader1, val_loader2))):
        # 0. clears all gradients.
        optimizer.zero_grad()
        val_loss = 0

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

        ########################
        # SEARCH VAL STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
        node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

        node_module_hist.append([node_idx_inv1, node_idx_inv2])

        edge_idx1, edge_idx_inv1 = edge_idx_to_batch(edge_idx_list1[idx_main], edge_idx_inv_list1[idx_main])
        edge_idx2, edge_idx_inv2 = edge_idx_to_batch(edge_idx_list2[idx_main], edge_idx_inv_list2[idx_main])

        edge_module_hist.append([edge_idx_inv1, edge_idx_inv2])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1, edge_idx=edge_idx1, edge_idx_inv=edge_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2, edge_idx=edge_idx2, edge_idx_inv=edge_idx_inv2)

        val_loss = (loss1 + loss2) / 2

        val_loss.backward()
        optimizer.step()
        val_loss_list.append(float(val_loss.data.cpu().numpy()))

      mean_loss = np.stack(val_loss_list).mean()
      logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(mean_loss, 0, epoch))
      self.writer.add_scalar('val_loss', mean_loss, iter_count)
      results['val_loss']+=[mean_loss]

      logger.info("Current Best Validation Loss = {}".format(best_val_loss))
      print(time.time() - epoch_time)
      print(asdf)

    with open(os.path.join(self.config.save_dir, 'node_module_hist.p'), "wb") as f:
      pickle.dump(node_module_hist, f)
      del node_module_hist

    with open(os.path.join(self.config.save_dir, 'edge_module_hist.p'), "wb") as f:
      pickle.dump(edge_module_hist, f)
      del edge_module_hist

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

    def propose_new_sturucture_batch(node_idx, node_idx_inv, edge_idx, edge_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        # change_node = False
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

      def propose_new_sturucture_edge(edge_idx, edge_idx_inv):
        change_edge = (np.random.rand() > 0.5)
        # change_edge = False
        if change_edge:
          idx = -1
          while idx == -1 or edge_idx_inv[idx] >= 2:
            idx = np.random.randint(len(edge_idx_inv))
          # Remove from old
          old_module = edge_idx_inv[idx]
          pos_in_old = edge_idx[old_module].index(idx)
          del edge_idx[old_module][pos_in_old]
          # Add to new
          new_module = old_module
          while new_module == old_module:
            new_module = np.random.randint(2)
          edge_idx_inv[idx] = new_module
          edge_idx[new_module].append(idx)
        return edge_idx, edge_idx_inv

      new_node_idx = []
      new_node_idx_inv = []
      new_edge_idx = []
      new_edge_idx_inv = []

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_sturucture(node_idx_batch, node_idx_inv_batch)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

      for edge_idx_batch, edge_idx_inv_batch in zip(edge_idx, edge_idx_inv):
        new_edge_idx_batch, new_edge_idx_inv_batch = propose_new_sturucture_edge(edge_idx_batch, edge_idx_inv_batch)
        new_edge_idx.append(new_edge_idx_batch)
        new_edge_idx_inv.append(new_edge_idx_inv_batch)

      return new_node_idx, new_node_idx_inv, new_edge_idx, new_edge_idx_inv

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    def edge_idx_to_batch(edge_idx, edge_idx_inv):
      batched_edge_idx = [[], []]
      num_edge_before = 0
      for idx, idx_inv in zip(edge_idx, edge_idx_inv):
        batched_edge_idx[0] += [__ + num_edge_before for __ in idx[0]]
        batched_edge_idx[1] += [__ + num_edge_before for __ in idx[1]]

        num_edge_before += len(idx_inv)
      batched_edge_idx_inv = []
      for _ in edge_idx_inv:
        batched_edge_idx_inv += _

      return batched_edge_idx, batched_edge_idx_inv

    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)
    tik = time.time()

    # create data loader
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, edge_module=True, random_init=self.model_conf.random_init)

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
    node_module_hist = []
    edge_module_hist = []

    temp = np.exp(self.initial_temp)

    node_idx_list = []
    node_idx_inv_list = []
    edge_idx_list = []
    edge_idx_inv_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])
      edge_idx_list.append(data['edge_idx'])
      edge_idx_inv_list.append(data['edge_idx_inv'])

    for step in tqdm(range(self.config.test.optim_step), desc="META TEST"):

      node_hist = copy.deepcopy(node_idx_inv_list)
      node_module_hist.append(node_hist)
      edge_hist = copy.deepcopy(edge_idx_inv_list)
      edge_module_hist.append(edge_hist)

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * step / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      loss_list = []
      for idx_main, data in tqdm(enumerate(test_loader)):
        loss_ = []
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])
        edge_idx, edge_idx_inv = edge_idx_to_batch(edge_idx_list[idx_main], edge_idx_inv_list[idx_main])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():
          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

          new_node_idx, new_node_idx_inv, new_edge_idx, new_edge_idx_inv = propose_new_sturucture_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main], edge_idx_list[idx_main], edge_idx_inv_list[idx_main])
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)
          new_edge_idx_, new_edge_idx_inv_ = edge_idx_to_batch(new_edge_idx, new_edge_idx_inv)

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_, edge_idx=new_edge_idx_, edge_idx_inv=new_edge_idx_inv_)

          updated_node_idx = []
          updated_node_idx_inv = []
          updated_edge_idx = []
          updated_edge_idx_inv = []
          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch))], loss_batch,
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
              updated_edge_idx.append(new_edge_idx[idx])
              updated_edge_idx_inv.append(new_edge_idx_inv[idx])
              loss_ += [float(new_loss.data.cpu().numpy())]
            else:
              updated_node_idx.append(node_idx_list[idx_main][idx])
              updated_node_idx_inv.append(node_idx_inv_list[idx_main][idx])
              updated_edge_idx.append(edge_idx_list[idx_main][idx])
              updated_edge_idx_inv.append(edge_idx_inv_list[idx_main][idx])
              loss_ += [float(old_loss.data.cpu().numpy())]

          node_idx_list[idx_main] = updated_node_idx
          node_idx_inv_list[idx_main] = updated_node_idx_inv
          edge_idx_list[idx_main] = updated_edge_idx
          edge_idx_inv_list[idx_main] = updated_edge_idx_inv

          test_loss = np.stack(loss_).mean()
          loss_list.append(test_loss)
      mean_loss = np.mean(loss_list)
      logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, mean_loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx],node_idx_inv_list[idx])
        edge_idx, edge_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])

        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "node_module_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(node_module_hist, f)

    file_name = os.path.join(self.config.save_dir, "edge_module_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(edge_module_hist, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return test_loss

#TRAIN NODE MODULE FROM VAL
class NeuralInferenceRunner_Meta3_zero(object):
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
    train_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    train_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)

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

    node_idx_list1 = []
    node_idx_inv_list1 = []
    node_idx_list2 = []
    node_idx_inv_list2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      node_idx_list1.append(data1['node_idx'])
      node_idx_list2.append(data2['node_idx'])

      node_idx_inv_list1.append(data1['node_idx_inv'])
      node_idx_inv_list2.append(data2['node_idx_inv'])


    for epoch in range(self.train_conf.max_epoch):
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      # ====================== training ============================= #
      model.eval()
      for idx_main, [data1, data2] in enumerate(zip(train_loader1, train_loader2)):
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
          # node_idx1, node_idx_inv1 = node_idx_to_batch(data1_val['node_idx'], data1_val['node_idx_inv'])
          # node_idx2, node_idx_inv2 = node_idx_to_batch(data2_val['node_idx'], data2_val['node_idx_inv'])
          node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
          node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

          _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
                                        target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
          _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
                                        target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

          loss = (loss1 + loss2) / 2

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info(
            "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

      model.train()
      lr_scheduler.step()
      val_loss_list = []
      # ===================== validation ============================ #
      for idx_main, [data1, data2] in tqdm(enumerate(zip(val_loader1, val_loader2))):
        # 0. clears all gradients.
        optimizer.zero_grad()
        val_loss = 0

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

        ########################
        # SEARCH VAL STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
        node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2)

        val_loss = (loss1 + loss2) / 2

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

      snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        epoch + 1,
        tag="final")

      logger.info("Current Best Validation Loss = {}".format(best_val_loss))

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
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, random_init=self.model_conf.random_init)

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
    loss_list = []

    temp = np.exp(self.initial_temp)

    node_idx_list = []
    node_idx_inv_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)
        loss_list += [x for x in loss_batch]
        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')

    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return test_loss

#TRAIN NODE,EDGE MODULE FROM VAL
class NeuralInferenceRunner_Meta4_zero(object):
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
    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    def edge_idx_to_batch(edge_idx, edge_idx_inv):
      batched_edge_idx = [[], []]
      num_edge_before = 0
      for idx, idx_inv in zip(edge_idx, edge_idx_inv):
        batched_edge_idx[0] += [__ + num_edge_before for __ in idx[0]]
        batched_edge_idx[1] += [__ + num_edge_before for __ in idx[1]]

        num_edge_before += len(idx_inv)

      batched_edge_idx_inv = []
      for _ in edge_idx_inv:
        batched_edge_idx_inv += _

      return batched_edge_idx, batched_edge_idx_inv

    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    train_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='train', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader1, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init)
    val_loader2, _, _ = eval(self.dataset_conf2.loader_name)(self.config2, split='val', shuffle=self.shuffle,parallel=self.parallel, master_node=self.master_node, edge_module=True, sort_by_number=self.dataset_conf2.data_path.split('/')[-1], random_init=self.model_conf.random_init)

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

    node_idx_list1 = []
    node_idx_inv_list1 = []
    node_idx_list2 = []
    node_idx_inv_list2 = []

    edge_idx_list1 = []
    edge_idx_inv_list1 = []
    edge_idx_list2 = []
    edge_idx_inv_list2 = []
    for data1, data2 in tqdm(zip(val_loader1, val_loader2)):
      node_idx_list1.append(data1['node_idx'])
      node_idx_list2.append(data2['node_idx'])

      node_idx_inv_list1.append(data1['node_idx_inv'])
      node_idx_inv_list2.append(data2['node_idx_inv'])

      edge_idx_list1.append(data1['edge_idx'])
      edge_idx_list2.append(data2['edge_idx'])

      edge_idx_inv_list1.append(data1['edge_idx_inv'])
      edge_idx_inv_list2.append(data2['edge_idx_inv'])

    for epoch in range(self.train_conf.max_epoch):
      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:
          temp *= self.temp_change
        else:
          temp = max(temp / self.temp_change, self.min_temp)

      # ====================== training ============================= #
      model.eval()
      for idx_main, [data1, data2, data1_val, data2_val] in enumerate(zip(train_loader1, train_loader2, val_loader1, val_loader2)):
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
          node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
          node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

          edge_idx1, edge_idx_inv1 = edge_idx_to_batch(edge_idx_list1[idx_main], edge_idx_inv_list1[idx_main])
          edge_idx2, edge_idx_inv2 = edge_idx_to_batch(edge_idx_list2[idx_main], edge_idx_inv_list2[idx_main])

          _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],
                                        target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1,
                                        edge_idx=edge_idx1, edge_idx_inv=edge_idx_inv1)
          _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'],
                                        target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2,
                                        edge_idx=edge_idx2, edge_idx_inv=edge_idx_inv2)

          loss = (loss1 + loss2) / 2

        train_loss = float(loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]
        self.writer.add_scalar('train_loss', train_loss, iter_count)

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info(
            "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

      model.train()
      lr_scheduler.step()
      val_loss_list = []
      # ===================== validation ============================ #
      for idx_main, [data1, data2] in tqdm(enumerate(zip(val_loader1, val_loader2))):
        # 0. clears all gradients.
        optimizer.zero_grad()
        val_loss = 0

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0,0]).contiguous().long()

        ########################
        # SEARCH VAL STRUCTURE
        ########################
        node_idx1, node_idx_inv1 = node_idx_to_batch(node_idx_list1[idx_main], node_idx_inv_list1[idx_main])
        node_idx2, node_idx_inv2 = node_idx_to_batch(node_idx_list2[idx_main], node_idx_inv_list2[idx_main])

        edge_idx1, edge_idx_inv1 = edge_idx_to_batch(edge_idx_list1[idx_main], edge_idx_inv_list1[idx_main])
        edge_idx2, edge_idx_inv2 = edge_idx_to_batch(edge_idx_list2[idx_main], edge_idx_inv_list2[idx_main])

        _, loss1, loss_batch1 = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'], target=data1['y'], node_idx=node_idx1, node_idx_inv=node_idx_inv1, edge_idx=edge_idx1, edge_idx_inv=edge_idx_inv1)
        _, loss2, loss_batch2 = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx2, node_idx_inv=node_idx_inv2, edge_idx=edge_idx2, edge_idx_inv=edge_idx_inv2)

        val_loss = (loss1 + loss2) / 2

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

      snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        epoch + 1,
        tag="final")

      logger.info("Current Best Validation Loss = {}".format(best_val_loss))

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

    def node_idx_to_batch(node_idx, node_idx_inv):
      batched_node_idx = [[], []]
      for i_batch, _ in enumerate(node_idx):
        batched_node_idx[0] += [__ + i_batch * self.dataset_conf.num_node for __ in _[0]]
        batched_node_idx[1] += [__ + i_batch * self.dataset_conf.num_node for __ in _[1]]
      batched_node_idx_inv = []
      for _ in node_idx_inv:
        batched_node_idx_inv += _
      return batched_node_idx, batched_node_idx_inv

    def edge_idx_to_batch(edge_idx, edge_idx_inv):
      batched_edge_idx = [[], []]
      num_edge_before = 0
      for idx, idx_inv in zip(edge_idx, edge_idx_inv):
        batched_edge_idx[0] += [__ + num_edge_before for __ in idx[0]]
        batched_edge_idx[1] += [__ + num_edge_before for __ in idx[1]]

        num_edge_before += len(idx_inv)
      batched_edge_idx_inv = []
      for _ in edge_idx_inv:
        batched_edge_idx_inv += _

      return batched_edge_idx, batched_edge_idx_inv

    print(self.dataset_conf.loader_name)
    print(self.dataset_conf.split)
    tik = time.time()

    # create data loader
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, edge_module=True, random_init=self.model_conf.random_init)

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

    temp = np.exp(self.initial_temp)

    node_idx_list = []
    node_idx_inv_list = []
    edge_idx_list = []
    edge_idx_inv_list = []
    loss_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])
      edge_idx_list.append(data['edge_idx'])
      edge_idx_inv_list.append(data['edge_idx_inv'])

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx],node_idx_inv_list[idx])
        edge_idx, edge_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])

        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv, edge_idx=edge_idx, edge_idx_inv=edge_idx_inv)

        loss_list += [x for x in loss_batch]
        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]


    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)
    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')
    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return test_loss

class Modular_meta_learning(object):
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

    self.initial_temp = self.train_conf.init_temp
    self.min_temp = 0.00001
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = self.train_conf.max_epoch
    self.temp_update = self.model_conf.temp

  @property
  def train(self):
    def propose_new_sturucture_batch(node_idx, node_idx_inv):
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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
    train_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init, meta_copy = self.train_conf.meta_copy)
    val_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init, meta_copy = self.train_conf.meta_copy)

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
    best_train_loss = np.inf
    results = defaultdict(list)
    temp = self.initial_temp

    node_module_hist = []

    node_idx_list = []
    node_idx_inv_list = []
    for data in tqdm(train_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])

    # num_data = len(node_idx_list)
    # node_idx_list = node_idx_list*self.train_conf.meta_copy
    # node_idx_inv_list = node_idx_inv_list*self.train_conf.meta_copy

    model.train()
    for epoch in range(self.train_conf.max_epoch):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate: #Reject가 많아지면 발생확률 up
          temp *= self.temp_change
        else:
          temp = temp / self.temp_change
          # temp = max(temp / self.temp_change, self.min_temp)

      #################################
      ####       RECORD HIST        ###
      #################################
      node_module_hist.append(copy.deepcopy(node_idx_list))
      results['temp_hist'] += [temp]
      results['SA_hist'] += [[acc_rate, self.SA_running_acc_rate, self.SA_running_factor]]

      train_loss_epoch = []
      val_loss_epoch = []
      true_reject_count = 0
      true_accept_count = 0
      false_accept_count = 0

      for idx_main, [data1, data2] in tqdm(enumerate(zip(train_loader, val_loader))):
        # for idx_copy in range(self.train_conf.meta_copy):
          structure_idx=idx_main
          loss = 0
          train_loss_batch = []
          val_loss_batch = []
          # structure_idx = idx_main + num_data * idx_copy

          ########################
          # SEARCH TRAIN STRUCTURE
          ########################
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
          ##############################
          # LOAD STRUCTURE FROM VAL DATA
          ##############################
          old_node_idx, old_node_idx_inv = copy.deepcopy(node_idx_list[structure_idx]), copy.deepcopy(node_idx_inv_list[structure_idx])
          node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[structure_idx], node_idx_inv_list[structure_idx])

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(node_idx_list[structure_idx],
                                                                        node_idx_inv_list[structure_idx])
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          model.eval()
          _, old_loss_train, old_loss_train_batch = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],target=data1['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)
          _, new_loss_train, new_loss_train_batch = model(data1['edge_attr'], data1['x'], data1['edge_index'],
                                                          data1['idx_msg_edge'], target=data1['y'],
                                                          node_idx=new_node_idx_, node_idx_inv=new_node_idx_inv_)

          model.train()
          _, old_loss_val, old_loss_val_batch = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)
          _, new_loss_val, new_loss_val_batch = model(data2['edge_attr'], data2['x'], data2['edge_index'],
                                                      data2['idx_msg_edge'], target=data2['y'], node_idx=new_node_idx_,
                                                      node_idx_inv=new_node_idx_inv_)

          ##############################
          # DECIDE STRUCTURE
          ##############################
          for idx, old_loss, new_loss in zip([_ for _ in range(len(old_loss_train_batch))], old_loss_train_batch, new_loss_train_batch):
            if self.temp_update:
              upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)

              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy():
                results['loss_change'] += [old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()]
                results['prob_accept'] += [prob_accept]
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept: #ACCEPT
              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy(): # update running frac of worse accepts
                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                self.SA_running_acc_rate = ((1 - upt_factor) * self.SA_running_acc_rate + upt_factor)  #ACC/FACTOR UP
                false_accept_count += 1
              else:
                true_accept_count += 1

              node_idx_list[structure_idx][idx] = new_node_idx[idx]
              node_idx_inv_list[structure_idx][idx] = new_node_idx_inv[idx]
              train_loss_batch += [float(new_loss_train.data.cpu().numpy())]
              val_loss_batch += [float(new_loss_val.data.cpu().numpy())]
              loss += new_loss_val_batch[idx]

            else: #REJECT
              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy(): # update running frac of worse accepts
                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                self.SA_running_acc_rate = (1 - upt_factor) * self.SA_running_acc_rate #ACC/FACTOR DOWN

              node_idx_list[structure_idx][idx] = old_node_idx[idx]
              node_idx_inv_list[structure_idx][idx] = old_node_idx_inv[idx]
              true_reject_count += 1
              train_loss_batch += [float(old_loss_train.data.cpu().numpy())]
              val_loss_batch += [float(old_loss_val.data.cpu().numpy())]
              loss += old_loss_val_batch[idx]

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          results['train_loss_batch'] += [np.stack(train_loss_batch).mean()]
          results['val_loss_batch'] += [np.stack(val_loss_batch).mean()]

          train_loss_epoch += [np.stack(train_loss_batch).mean()]
          val_loss_epoch += [np.stack(val_loss_batch).mean()]

      results['true_accept'] += [true_accept_count]
      results['false_accept'] += [false_accept_count]
      results['true_reject'] += [true_reject_count]

      results['train_loss'] += [np.stack(train_loss_epoch).mean()]
      results['val_loss'] += [np.stack(val_loss_epoch).mean()]

      mean_loss = np.stack(train_loss_epoch).mean()
      mean_loss_val = np.stack(val_loss_epoch).mean()

      # save best model
      if mean_loss < best_train_loss:
        best_train_loss = mean_loss
        snapshot(
          model.module if self.use_gpu else model,
          optimizer,
          self.config,
          epoch + 1,
          tag="best")

      snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        epoch + 1,
        tag="final")

      logger.info("Train/Val Loss @ epoch {:04d}  = {}".format(epoch + 1, mean_loss, mean_loss_val))
      logger.info("Current Best Train Loss = {}".format(best_train_loss))

    with open(os.path.join(self.config.save_dir, 'node_module_hist.p'), "wb") as f:
      pickle.dump(node_module_hist, f)
      del node_module_hist

    print(np.array(results['hidden_state']).shape)
    results['best_train_loss'] += [best_train_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Train Loss = {}".format(best_train_loss))

    return best_train_loss

  def test(self):
    def propose_new_sturucture_batch(node_idx, node_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        # change_node = (np.random.rand() > 0.5)
        change_node = True
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, random_init=self.model_conf.random_init)

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
    pred_pts = []
    gt_pts = []
    node_module_hist = []
    loss_list = []

    node_idx_list = []
    node_idx_inv_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])

    for step in tqdm(range(self.config.test.optim_step), desc="META TEST"):

      node_hist = copy.deepcopy(node_idx_inv_list)
      node_module_hist.append(node_hist)

      loss_ = []
      for idx_main, data in tqdm(enumerate(test_loader)):
        old_node_idx, old_node_idx_inv = copy.deepcopy(node_idx_list[idx_main]), copy.deepcopy(node_idx_inv_list[idx_main])
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():

          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch))], loss_batch,
                                             new_loss_batch):

            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              node_idx_list[idx_main][idx] = new_node_idx[idx]
              node_idx_inv_list[idx_main][idx] = new_node_idx_inv[idx]
              loss_ += [float(new_loss.data.cpu().numpy())]
            else:
              node_idx_list[idx_main][idx] = old_node_idx[idx]
              node_idx_inv_list[idx_main][idx] = old_node_idx_inv[idx]
              loss_ += [float(old_loss.data.cpu().numpy())]

      mean_loss = np.stack(loss_).mean()
      loss_list.append(mean_loss)
      logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, mean_loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)

    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')

    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "node_module_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(node_module_hist, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return loss_list

class Modular_meta_learning_edge(object):
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

    self.initial_temp = self.train_conf.init_temp
    self.min_temp = 0.00001
    self.temp_change = 1.1
    self.SA_running_acc_rate = 1e-9
    self.SA_running_factor = 1e-9
    self.initial_acc = 0
    self.temp_slope_opt_steps = self.train_conf.max_epoch
    self.temp_update = self.model_conf.temp

  @property
  def train(self):
    def propose_new_sturucture_batch(node_idx, node_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        change_node = (np.random.rand() > 0.5)
        # change_node = False
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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

    def edge_idx_to_batch(edge_idx, edge_idx_inv):
      batched_edge_idx = [[], []]
      num_edge_before = 0
      for idx, idx_inv in zip(edge_idx, edge_idx_inv):
        batched_edge_idx[0] += [__ + num_edge_before for __ in idx[0]]
        batched_edge_idx[1] += [__ + num_edge_before for __ in idx[1]]
        num_edge_before += len(idx_inv)
      batched_edge_idx_inv = []
      for _ in edge_idx_inv:
        batched_edge_idx_inv += _
      return batched_edge_idx, batched_edge_idx_inv

    train_begin_time = time.time()
    # create data loader
    torch.cuda.empty_cache()
    train_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init, meta_copy = self.train_conf.meta_copy)
    val_loader, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle, parallel=self.parallel, master_node=self.master_node, sort_by_number=self.dataset_conf.data_path.split('/')[-1], random_init=self.model_conf.random_init, meta_copy = self.train_conf.meta_copy)

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
    best_train_loss = np.inf
    results = defaultdict(list)
    temp = self.initial_temp

    node_module_hist = []

    node_idx_list = []
    node_idx_inv_list = []
    edge_idx_list = []
    edge_idx_inv_list = []
    for data in tqdm(train_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])
      edge_idx_list.append(data['edge_idx'])
      edge_idx_inv_list.append(data['edge_idx_inv'])

    # num_data = len(node_idx_list)
    # node_idx_list = node_idx_list*self.train_conf.meta_copy
    # node_idx_inv_list = node_idx_inv_list*self.train_conf.meta_copy

    model.train()
    for epoch in range(self.train_conf.max_epoch):

      if self.temp_update:
        acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
        if self.SA_running_acc_rate / self.SA_running_factor < acc_rate: #Reject가 많아지면 발생확률 up
          temp *= self.temp_change
        else:
          # temp = max(temp / self.temp_change, self.min_temp)
          temp = temp / self.temp_change

      #################################
      ####       RECORD HIST        ###
      #################################
      node_module_hist.append(copy.deepcopy(node_idx_list))
      results['temp_hist'] += [temp]
      results['SA_hist'] += [[acc_rate, self.SA_running_acc_rate, self.SA_running_factor]]

      train_loss_epoch = []
      val_loss_epoch = []
      true_reject_count = 0
      true_accept_count = 0
      false_accept_count = 0

      for idx_main, [data1, data2] in tqdm(enumerate(zip(train_loader, val_loader))):
          structure_idx=idx_main
          loss = 0
          train_loss_batch = []
          val_loss_batch = []

          ########################
          # SEARCH TRAIN STRUCTURE
          ########################
          if "TorchGNN_MsgGNN" not in self.model_conf.name:
            data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
            data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
          ##############################
          # LOAD STRUCTURE FROM VAL DATA
          ##############################
          node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[structure_idx], node_idx_inv_list[structure_idx])
          edge_idx, edge_idx_inv = edge_idx_to_batch(edge_idx_list[structure_idx], edge_idx_inv_list[structure_idx])

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(node_idx_list[structure_idx],
                                                                        node_idx_inv_list[structure_idx])
          new_edge_idx, new_edge_idx_inv = update_edge()


          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          model.eval()
          _, old_loss_train, old_loss_train_batch = model(data1['edge_attr'], data1['x'], data1['edge_index'], data1['idx_msg_edge'],target=data1['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)
          _, new_loss_train, new_loss_train_batch = model(data1['edge_attr'], data1['x'], data1['edge_index'],
                                                          data1['idx_msg_edge'], target=data1['y'],
                                                          node_idx=new_node_idx_, node_idx_inv=new_node_idx_inv_)

          model.train()
          _, old_loss_val, old_loss_val_batch = model(data2['edge_attr'], data2['x'], data2['edge_index'], data2['idx_msg_edge'], target=data2['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)
          _, new_loss_val, new_loss_val_batch = model(data2['edge_attr'], data2['x'], data2['edge_index'],
                                                      data2['idx_msg_edge'], target=data2['y'], node_idx=new_node_idx_,
                                                      node_idx_inv=new_node_idx_inv_)

          ##############################
          # DECIDE STRUCTURE
          ##############################
          for idx, old_loss, new_loss in zip([_ for _ in range(len(old_loss_train_batch))], old_loss_train_batch, new_loss_train_batch):
            if self.temp_update:
              upt_factor = min(0.0001, self.SA_running_acc_rate / self.SA_running_factor)
              prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)

              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy():
                results['loss_change'] += [old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()]
                results['prob_accept'] += [prob_accept]
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() or np.random.rand() < prob_accept
            else:
              accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept: #ACCEPT
              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy(): # update running frac of worse accepts
                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                self.SA_running_acc_rate = ((1 - upt_factor) * self.SA_running_acc_rate + upt_factor)  #ACC/FACTOR UP
                false_accept_count += 1
              else:
                true_accept_count += 1

              node_idx_list[structure_idx][idx] = new_node_idx[idx]
              node_idx_inv_list[structure_idx][idx] = new_node_idx_inv[idx]
              train_loss_batch += [float(new_loss_train.data.cpu().numpy())]
              val_loss_batch += [float(new_loss_val.data.cpu().numpy())]
              loss += new_loss_val_batch[idx]

            else: #REJECT
              if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy(): # update running frac of worse accepts
                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                self.SA_running_acc_rate = (1 - upt_factor) * self.SA_running_acc_rate #ACC/FACTOR DOWN

              true_reject_count += 1
              train_loss_batch += [float(old_loss_train.data.cpu().numpy())]
              val_loss_batch += [float(old_loss_val.data.cpu().numpy())]
              loss += old_loss_val_batch[idx]

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          results['train_loss_batch'] += [np.stack(train_loss_batch).mean()]
          results['val_loss_batch'] += [np.stack(val_loss_batch).mean()]

          train_loss_epoch += [np.stack(train_loss_batch).mean()]
          val_loss_epoch += [np.stack(val_loss_batch).mean()]

      results['true_accept'] += [true_accept_count]
      results['false_accept'] += [false_accept_count]
      results['true_reject'] += [true_reject_count]

      results['train_loss'] += [np.stack(train_loss_epoch).mean()]
      results['val_loss'] += [np.stack(val_loss_epoch).mean()]

      mean_loss = np.stack(train_loss_epoch).mean()
      mean_loss_val = np.stack(val_loss_epoch).mean()

      # save best model
      if mean_loss < best_train_loss:
        best_train_loss = mean_loss
        snapshot(
          model.module if self.use_gpu else model,
          optimizer,
          self.config,
          epoch + 1,
          tag="best")

      snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        epoch + 1,
        tag="final")

      logger.info("Train/Val Loss @ epoch {:04d}  = {}".format(epoch + 1, mean_loss, mean_loss_val))
      logger.info("Current Best Train Loss = {}".format(best_train_loss))

    with open(os.path.join(self.config.save_dir, 'node_module_hist.p'), "wb") as f:
      pickle.dump(node_module_hist, f)
      del node_module_hist

    print(np.array(results['hidden_state']).shape)
    results['best_train_loss'] += [best_train_loss]
    train_end_time = time.time()
    results['total_time'] = train_end_time-train_begin_time
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Train Loss = {}".format(best_train_loss))

    return best_train_loss

  def test(self):
    def propose_new_sturucture_batch(node_idx, node_idx_inv):
      def propose_new_sturucture(node_idx, node_idx_inv):
        # change_node = (np.random.rand() > 0.5)
        change_node = True
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

      for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
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
    test_loader, name_list, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test', shuffle=False, random_init=self.model_conf.random_init)

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
    pred_pts = []
    gt_pts = []
    node_module_hist = []
    loss_list = []

    node_idx_list = []
    node_idx_inv_list = []
    for data in tqdm(test_loader):
      node_idx_list.append(data['node_idx'])
      node_idx_inv_list.append(data['node_idx_inv'])

    for step in tqdm(range(self.config.test.optim_step), desc="META TEST"):

      node_hist = copy.deepcopy(node_idx_inv_list)
      node_module_hist.append(node_hist)

      loss_ = []
      for idx_main, data in tqdm(enumerate(test_loader)):
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])

        if "TorchGNN_MsgGNN" not in self.model_conf.name:
          data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

        with torch.no_grad():

          log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                             target=data['y'], node_idx=node_idx, node_idx_inv=node_idx_inv)

          new_node_idx, new_node_idx_inv = propose_new_sturucture_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main])
          new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv)

          log_prob_new, loss_new, new_loss_batch = model(data['edge_attr'], data['x'], data['edge_index'],
                                                         data['idx_msg_edge'],
                                                         target=data['y'], node_idx=new_node_idx_,
                                                         node_idx_inv=new_node_idx_inv_)

          for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch))], loss_batch,
                                             new_loss_batch):

            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

            if accept:
              node_idx_list[idx_main][idx] = new_node_idx[idx]
              node_idx_inv_list[idx_main][idx] = new_node_idx_inv[idx]
              loss_ += [float(new_loss.data.cpu().numpy())]
            else:
              loss_ += [float(old_loss.data.cpu().numpy())]

      mean_loss = np.stack(loss_).mean()
      loss_list.append(mean_loss)
      logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, mean_loss))

    print("=======================================")
    print("TEST")
    print("=======================================")
    for idx, data in tqdm(enumerate(test_loader)):
      if "TorchGNN_MsgGNN" not in self.model_conf.name:
        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

      with torch.no_grad():
        node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx], node_idx_inv_list[idx])
        log_prob, loss, loss_batch = model(data['edge_attr'], data['x'], data['edge_index'], data['idx_msg_edge'],
                                           target=data['y'],
                                           node_idx=node_idx, node_idx_inv=node_idx_inv)

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['y'].data.cpu().numpy()]

    pred_pts = np.concatenate(pred_pts, axis=0)
    gt_pts = np.concatenate(gt_pts, axis=0)
    name_list = np.array(name_list)

    np.savetxt(self.config.save_dir + '/pred_pts_' + self.dataset_conf.split + '.csv', pred_pts, delimiter='\t')
    np.savetxt(self.config.save_dir + '/gt_pts_' + self.dataset_conf.split + '.csv', gt_pts, delimiter='\t')

    total_time = time.time() - tik

    with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
      pickle.dump(total_time, f)

    file_name = os.path.join(self.config.save_dir, "name.p")
    with open(file_name, 'wb') as f:
      pickle.dump(name_list, f)

    file_name = os.path.join(self.config.save_dir, "file.p")
    with open(file_name, 'wb') as f:
      pickle.dump(file_list, f)

    file_name = os.path.join(self.config.save_dir, "node_module_hist.p")
    with open(file_name, 'wb') as f:
      pickle.dump(node_module_hist, f)

    file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
    with open(file_name, 'wb') as f:
      pickle.dump(loss_list, f)

    return loss_list

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

    return test_los