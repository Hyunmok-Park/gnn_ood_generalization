import os
import yaml
import time
import argparse
from easydict import EasyDict as edict


def parse_arguments():
  parser = argparse.ArgumentParser(description="Running Experiments of GNN")
  parser.add_argument(
      '-c',
      '--config_file',
      type=str,
      default="config/resnet101_cifar.json",
      required=True,
      help="Path of config file")
  parser.add_argument(
      '-l',
      '--log_level',
      type=str,
      default='INFO',
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument('-m', '--comment', help="Experiment comment")
  parser.add_argument('-t', '--test', help="Test model", action='store_true')
  args = parser.parse_args()
  return args


def get_config(config_file, sample_id, exp_dir=None):
  """ Construct and snapshot hyper parameters """
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

  # create hyper parameters
  config.run_id = str(os.getpid())
  if config.model.degree_emb == True:
      config.exp_name = '_'.join([
          config.model.name, "NODE", sample_id, config.dataset.name,
          time.strftime('%Y-%b-%d-%H-%M-%S'), str(config.dataset.split), str(config.dataset.num_node), config.dataset.data_path.split('/')[-1], str(config.model.hidden_dim), str(config.model.num_prop), config.model.aggregate_type
      ])
  else:
      config.exp_name = '_'.join([
          config.model.name, sample_id, config.dataset.name, config.dataset.data_path.split('/')[-1],
          time.strftime('%Y-%b-%d-%H-%M-%S'), str(config.dataset.split), str(config.model.hidden_dim), str(config.model.num_prop),
          config.model.aggregate_type, "JK" if config.model.jumping else "", "SK" if config.model.skip_connection else "", "IP" if config.model.interpol else "", "MN_2" if config.model.master_node else "","MASK_{}".format(config.model.masking_number) if config.model.masking else "",
          "PRETEXT" if config.model.train_pretext else ""
      ])
  if exp_dir is not None:
    config.exp_dir = exp_dir

  config.save_dir = os.path.join(config.exp_dir, config.exp_name)

  # snapshot hyperparameters
  mkdir(config.exp_dir)
  mkdir(config.save_dir)

  save_name = os.path.join(config.save_dir, 'config.yaml')
  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)
