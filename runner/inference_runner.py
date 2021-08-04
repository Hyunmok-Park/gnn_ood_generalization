from __future__ import (division, print_function)

import collections
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
__all__ = ['NeuralInferenceRunner', 'AlgorithmicInferenceRunner_bp', 'AlgorithmicInferenceRunner', 'NeuralInferenceRunner_MT']

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