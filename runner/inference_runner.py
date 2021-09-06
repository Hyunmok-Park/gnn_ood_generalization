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
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.topology import NetworkTopology, get_msg_graph

logger = get_logger('exp_logger')
EPS = float(np.finfo(np.float32).eps)
__all__ = ['NeuralInferenceRunner', 'AlgorithmicInferenceRunner', 'AlgorithmicInferenceRunner_bp']


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

  def train(self):
    train_begin_time = time.time()
    # create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
    val_dataset = eval(self.dataset_conf.loader_name)(self.config, split='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=False,
        num_workers=self.train_conf.num_workers,
        collate_fn=val_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    if self.use_gpu:
      print("Using GPU dataparallel")
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

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
      load_model(model, self.train_conf.resume_model, optimizer=optimizer)

    #========================= Training Loop =============================#
    iter_count = 0
    best_val_loss = np.inf
    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):
      # ===================== validation ============================ #
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
        model.eval()
        val_loss = []
        # val_norm_loss = []

        for data in tqdm(val_loader):
          if self.use_gpu:
            # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'] = data_to_gpu(
            # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'])

            data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'] = data_to_gpu(
            data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'])

            # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'] = data_to_gpu(
            # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'])


          with torch.no_grad():
            # _, loss = model(data['J'], data['b'], data['msg_node'], data['msg_adj'], target=data['prob_gt'])
            _, loss, norm_loss, _ = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], data['degree'], target=data['prob_gt'])
            # _, loss, __ = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], target=data['prob_gt'])
            # _, loss = model(data['J_msg'], data['b'], data['msg_node'],data['idx_msg_edge'], target = data['bp_step'])

          val_loss += [float(loss.data.cpu().numpy())]
          # val_norm_loss += [float(norm_loss.data.cpu().numpy())]

        val_loss = np.stack(val_loss).mean()
        # val_norm_loss = np.stack(val_norm_loss).mean()
        results['val_loss'] += [val_loss]
        # results['val_norm_loss'] += [val_norm_loss]
        logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))
        # logger.info("Avg. Norm validation Loss = {} +- {}".format(val_norm_loss, 0))
        self.writer.add_scalar('val_loss', val_loss, iter_count)
        # self.writer.add_scalar('val_norm_loss', val_norm_loss, iter_count)

        # save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            epoch + 1,
            tag='best')

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
        if self.use_gpu:
          # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'] = data_to_gpu(
          # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'])
          data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'] = data_to_gpu(
          data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'])
          # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'] = data_to_gpu(
          # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'])


        # 1. forward pass
        # 2. compute loss
        # _, loss = model(data['J'], data['b'], data['msg_node'], data['msg_adj'], target=data['prob_gt'])
        _, loss, norm_loss, hidden_state = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], data['degree'], target=data['prob_gt'])
        # _, loss, count = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], target=data['prob_gt'])
        # _, loss = model(data['J_msg'], data['b'], data['msg_node'],data['idx_msg_edge'], target = data['bp_step'])

        # 3. backward pass (accumulates gradients).
        # loss2 = loss + 0.00001 * norm_loss
        # loss2.backward()
        loss.backward()
        # 4. performs a single update step.
        optimizer.step()

        train_loss = float(loss.data.cpu().numpy())
        # norm_loss = float(norm_loss.data.cpu().numpy())
        results['train_loss'] += [train_loss]
        # results['norm_loss'] += [norm_loss]
        results['train_step'] += [iter_count]
        # results['hidden_state'] += [hidden_state]
        #results['count'] += [count]
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
    test_dataset = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split)
    # create data loader
    test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=self.test_conf.batch_size,
      shuffle=False,
      num_workers=self.test_conf.num_workers,
      collate_fn=test_dataset.collate_fn,
      drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)
    if 'GNN' in self.model_conf.name:
      load_model(model, self.test_conf.test_model)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    model.eval()
    test_loss = []
    pred_pts = []
    gt_pts = []
    name_list = []
    state_hist = []
    tik_end = time.time()
    for data in tqdm(test_loader):
      if self.use_gpu:
        # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'] = data_to_gpu(
        # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['prob_gt'])

        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'] = data_to_gpu(
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['degree'])

        # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'] = data_to_gpu(
        # data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['bp_step'])

        # data['J'], data['b'], data['msg_adj'], data['msg_node'], data['msg_adj'] = data_to_gpu(
        #   data['J'], data['b'], data['msg_adj'], data['msg_node'], data['msg_adj'])
        # bp_step = []
        # num_node = int(len(data['J']) / self.test_conf.batch_size)
        # for i in range(self.test_conf.batch_size):
        #   J = data['J'][i * num_node:(i + 1) * num_node, :]
        #   b = data['b'][i * num_node:(i + 1) * num_node, :]
        #   msg_node = data['msg_node'][i * int(len(data['msg_node']) / self.test_conf.batch_size):(i + 1) * int(
        #     len(data['msg_node']) / self.test_conf.batch_size), :]
        #   msg_adj = data['msg_adj'].numpy()
        #   bp_prob, bp_step_ = BeliefPropagation(J, b, msg_node, msg_adj).inference(
        #     max_iter=15, damping=0.0, observe_mask=None)
        #   bp_step += [bp_step_]
        # bp_step = np.concatenate(bp_step, axis=0)
        # data['bp_step'] = bp_step


      with torch.no_grad():
        # log_prob, loss = model(data['J'], data['b'], data['msg_node'], data['msg_adj'], target=data['prob_gt'])
        log_prob, loss, norm_loss, hist = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], data['degree'], target=data['prob_gt'])
        # log_prob, loss, count = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], target=data['prob_gt'])

        # bp_step = []
        # num_node = int(len(data['J']) / self.test_conf.batch_size)
        # for i in range(self.test_conf.batch_size):
        #   J = data['J'][i * num_node:(i + 1) * num_node, :]
        #   b = data['b'][i * num_node:(i + 1) * num_node, :]
        #   msg_node = data['msg_node'][i * int(len(data['msg_node'])/self.test_conf.batch_size):(i + 1) * int(len(data['msg_node'])/self.test_conf.batch_size), :]
        #   print(msg_node.shape)
        #   msg_adj = data['msg_adj'].numpy()
        #   bp_prob, bp_step_ = BeliefPropagation(J, b, msg_node, msg_adj).inference(
        #     max_iter=15, damping=0.0, observe_mask=None)
        #   bp_step += [bp_step_]
        # bp_step = np.concatenate(bp_step, axis=0)
        # data['bp_step'] = torch.from_numpy(bp_step)
        # y_step_, loss = model(data['J_msg'], data['b'], data['msg_node'],data['idx_msg_edge'], target = data['bp_step'])

        loss = 0 if loss < 0 else float(loss.data.cpu().numpy())
        test_loss += [loss]
        state_hist += [hist]

        pred_pts += [torch.exp(log_prob).data.cpu().numpy()]
        gt_pts += [data['prob_gt'].data.cpu().numpy()]

        name_list.append(data['name'])
        # name_list.append(data['file_name'])

        # y_step += [y_step_]
        # bp_step2 += [data['bp_step'].data.cpu().numpy()]

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

    # y_step = np.concatenate(y_step, axis=0)
    # bp_step2 = np.concatenate(bp_step2, axis=0)
    # np.savetxt(self.config.save_dir + '/y_step_' + self.dataset_conf.split + '.csv', y_step, delimiter='\t')
    # np.savetxt(self.config.save_dir + '/bp_step_' + self.dataset_conf.split + '.csv', bp_step2, delimiter='\t')

    return test_loss



# python3 run_exp.py -c config/bp.yaml -t
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
    test_dataset = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split)
    # create data loader
    test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=self.test_conf.batch_size,
      shuffle=False,
      num_workers=self.test_conf.num_workers,
      collate_fn=test_dataset.collate_fn,
      drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)
    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # loss_dict = defaultdict(list)
    # marginal_gt_dict = defaultdict(list)
    # marginal_pred_dict = defaultdict(list)

    loss_dict = []
    marginal_gt_dict = []
    marginal_pred_dict = []
    prob_step = []

    model.eval()
    for idx, data in tqdm(enumerate(test_loader)):
      if self.use_gpu:
        # data['J'], data['b'], data['adj'], data['msg_adj'], data['msg_node'], data['prob_gt'] = data_to_gpu(
        # data['J'], data['b'], data['adj'], data['msg_adj'], data['msg_node'], data['prob_gt'])
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['msg_adj'] = data_to_gpu(
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'], data['msg_adj'])

      with torch.no_grad():
        prob_pred, loss, prob_step_ = model(data['J'], data['b'], data['msg_node'], data['msg_adj'], target=data['prob_gt'])
        # prob_pred, loss = model(data['J_msg'], data['b'], data['msg_node'], data['idx_msg_edge'], target=data['prob_gt'])

        loss = 0 if loss < 0 else loss.data.cpu().numpy()
        # topology = '_'
        # loss_dict[topology] += [loss]
        # marginal_gt_dict[topology] += [data['prob_gt'].data.cpu().numpy()]
        # marginal_pred_dict[topology] += [prob_pred.data.cpu().numpy()]
        #
        # if idx % self.num_graphs == 0:
        #   test_loss = -np.log10(np.stack(loss_dict[topology]) + EPS)
        #   avg_test_loss = float(np.mean(test_loss))
        #   std_test_loss = float(np.std(test_loss))
        #   logger.info("{} Test({}) -log10 KL-div = {} +- {}".format(self.model_conf.name, topology, avg_test_loss, std_test_loss))
        #
        #   pred_pts = np.concatenate(marginal_pred_dict[topology], axis=0)
        #   gt_pts = np.concatenate(marginal_gt_dict[topology], axis=0)
        #
        #   file_name = os.path.join(self.save_dir, 'gt_{}.csv'.format(topology))
        #   np.savetxt(file_name, gt_pts, delimiter='\t')
        #   file_name = os.path.join(self.save_dir, 'pred_{}.csv'.format(topology))
        #   np.savetxt(file_name, pred_pts, delimiter='\t')

        loss_dict += [loss]
        marginal_gt_dict += [data['prob_gt'].data.cpu().numpy()]
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

    # for tp in loss_dict:
    #   print('{}: {}'.format(tp, float(np.mean(-np.log10(np.stack(loss_dict[tp]) + EPS)))))

    return loss_dict

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
    test_dataset = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split)
    # create data loader
    test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=self.test_conf.batch_size,
      shuffle=False,
      num_workers=self.test_conf.num_workers,
      collate_fn=test_dataset.collate_fn,
      drop_last=False)

    # create models
    # model = eval(self.model_conf.name)(self.config)
    # if self.use_gpu:
    #   model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # loss_dict = defaultdict(list)
    # marginal_gt_dict = defaultdict(list)
    # marginal_pred_dict = defaultdict(list)

    loss_dict = []
    marginal_gt_dict = []
    marginal_pred_dict = []
    prob_step = []

    # model.eval()
    for idx, data in tqdm(enumerate(test_loader)):
      if self.use_gpu:
        # data['J'], data['b'], data['adj'], data['msg_adj'], data['msg_node'], data['prob_gt'] = data_to_gpu(
        # data['J'], data['b'], data['adj'], data['msg_adj'], data['msg_node'], data['prob_gt'])
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'] = data_to_gpu(
        data['J_msg'], data['b'], data['idx_msg_edge'], data['msg_node'], data['prob_gt'])

      with torch.no_grad():
        msg_node, msg_adj = get_msg_graph(nx.from_numpy_matrix(data['J'].numpy()))
        prob_pred, prob_step_ = BeliefPropagation(data['J'], data['b'], msg_node, msg_adj).inference(max_iter=15, damping=0.0,
                                                                                        observe_mask=None)

        # loss = 0 if loss < 0 else loss.data.cpu().numpy()

        # loss_dict += [loss]
        marginal_gt_dict += [data['prob_gt'].data.cpu().numpy()]
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

    return loss_dict

# Non-torch-based BP
# class AlgorithmicInferenceRunner(object):
#
#   def __init__(self, config):
#     self.config = config
#     self.dataset_conf = config.dataset
#     self.model_conf = config.model
#     self.train_conf = config.train
#     self.test_conf = config.test
#     self.use_gpu = config.use_gpu
#     self.gpus = config.gpus
#     self.save_dir = config.save_dir
#     self.num_graphs = 10
#
#     self.max_iter = config.model.max_iter
#     self.kl_order = config.model.kl_order
#     self.damping = config.model.damping
#
#
#   def test(self):
#     test_dataset = eval(self.dataset_conf.loader_name)(self.config, split=self.dataset_conf.split)
#     # create data loader
#     test_loader = torch.utils.data.DataLoader(
#       test_dataset,
#       batch_size=self.test_conf.batch_size,
#       shuffle=False,
#       num_workers=self.test_conf.num_workers,
#       collate_fn=test_dataset.collate_fn,
#       drop_last=False)
#
#     loss_dict = defaultdict(list)
#     marginal_gt_dict = defaultdict(list)
#     marginal_pred_dict = defaultdict(list)
#
#     for idx, data in tqdm(enumerate(test_loader)):
#       model = BeliefPropagation(data['J'].numpy(), data['b'].numpy(), data['msg_node'], data['msg_adj'].numpy())
#       prob, loss = model.inference(self.max_iter, self.kl_order, self.damping, target = data['prob_gt'])
#       prob, loss = torch.tensor(prob), torch.tensor(loss)
#
#       topology = data['topology']
#       loss_dict[topology] += [loss.data.cpu().numpy()]
#       marginal_gt_dict[topology] += [data['prob_gt'].data.cpu().numpy()]
#       marginal_pred_dict[topology] += [prob.data.cpu().numpy()]
#
#       if (1 + idx) % self.num_graphs == 0:
#         test_loss = -np.log10(np.stack(loss_dict[topology]) + EPS)
#         avg_test_loss = float(np.mean(test_loss))
#         std_test_loss = float(np.std(test_loss))
#         logger.info("Test -log10 KL-div = {} +- {}".format(avg_test_loss, std_test_loss))
#
#         pred_pts = np.concatenate(marginal_pred_dict[topology], axis=0)
#         gt_pts = np.concatenate(marginal_gt_dict[topology], axis=0)
#
#         file_name = os.path.join(self.save_dir, 'gt_{}.csv'.format(topology))
#         np.savetxt(file_name, gt_pts, delimiter='\t')
#         file_name = os.path.join(self.save_dir, 'pred_{}.csv'.format(topology))
#         np.savetxt(file_name, pred_pts, delimiter='\t')
#
#
#     for xx in loss_dict:
#       print('{}: {}'.format(xx, float(np.mean(-np.log10(np.stack(loss_dict[xx]) + EPS)))))
#
#     return loss_dict

