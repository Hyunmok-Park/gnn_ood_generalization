import numpy as np
import yaml


class HyperGrid(object):
  """ Generate Hyper-parameter Grid """

  def __init__(self, key, val, bind_key=None):
    """
      key: list, hyper-parameter name, e.g., ["learning_rate", "momentum"]
      val: list, hyper-parameter values, e.g., [[0.01, 0.001], [0.9, 0.95]]
      bind_key: list, indicates which set of parameters are bound together, 
          e.g., [["learning_rate", "momentum"]]
          N.B., vals of the bound keys should be same-length list

      Usage 1:
        # loop over all possible combinations
        # example below will try all pairs of learning rate and momentum
        # i.e., (0.01, 0.9), (0.001, 0.9), (0.01, 0.95), (0.001, 0.95)

        key = ["learning_rate", "momentum"]
        val = [[0.01, 0.001], [0.9, 0.95]]
        bind_key = None

      Usage 2:
        # loop over particular combinations
        # example below will try bind pairs of learning rate and momentum
        # i.e., (0.01, 0.9), (0.001, 0.95)

        key = ["learning_rate", "momentum"]
        val = [[0.01, 0.001], [0.9, 0.95]]
        bind_key = [["learning_rate", "momentum"]]
    """
    self._para_key = key
    self._para_val = val
    self._bind_key = bind_key
    self._key_dict = dict(zip(key, range(len(key))))

    # check input 
    if len(key) != len(val):
      raise ValueError("Key and Val should have the same length!")

    # check bind parameters
    bind_idx = []

    if self._bind_key:
      for key_list in bind_key:
        idx = [self._key_dict[kk] for kk in key_list]
        bind_idx.append(idx)

        len_list = np.array([len(val[ii]) for ii in idx])
        len_list = (len_list - len_list[0]).astype(np.bool)

        if len_list.any():
          raise ValueError("lengths of bound key values are not equal!")

      self._bind_idx = bind_idx

  def gen_grid(self):

    idx_all = np.meshgrid(*[np.arange(len(xx)) for xx in self._para_val])
    idx_all = [xx.flatten() for xx in idx_all]

    if self._bind_key is not None:
      idx_keep = []
      for idx in self._bind_idx:
        array = np.concatenate(
            [np.expand_dims(
                idx_all[jj], axis=1) for jj in idx], axis=1)
        idx_keep += [
            np.logical_not((array - np.expand_dims(
                array[:, 0], axis=1)).astype(np.bool)).all(axis=1,
                                                           keepdims=True)
        ]

      idx_keep = np.concatenate(idx_keep, axis=1)
      idx_keep = idx_keep.all(axis=1)
      idx_all = [xx[idx_keep] for xx in idx_all]

    return idx_all


# unit test
# def test():
#   key = ["lr", "alpha", "layer", "step"]
#   val = [[0.1, 0.01], [0.9, 0.95, 1.0], [1, 2, 3], [1, 2, 3]]
#   bind_key = [["alpha", "layer", "step"], ["layer", "step"]]
#
#   grid = HyperGrid(key, val, bind_key=None)
#   idx_all = grid.gen_grid()
#   num_trial = len(idx_all[0])
#
#   for ii in range(num_trial):
#     val_list = [val[jj][idx_all[jj][ii]] for jj in range(len(key))]
#     print("lr = {} || alpha = {} || layer = {} || step = {}".format(val_list[
#         0], val_list[1], val_list[2], val_list[3]))
def test():
  key = ["loss", "hidden_dim", "num_prop", "aggregate_type", "lr", "wd"]
  val = [['MSE', 'KL-pq', 'KL-qp'], [64, 128], [5, 10], ['att', 'avg', 'sum'], [0.001], [0]]
  # val = [['MSE', 'KL-pq'], [64], [10], ['avg', 'sum'], [0.001], [0]]
  bind_key = [["lr", "wd"]]

  grid = HyperGrid(key, val, bind_key=bind_key)
  idx_all = grid.gen_grid()
  num_trial = len(idx_all[0])

  for ii in range(num_trial):
    val_list = [val[jj][idx_all[jj][ii]] for jj in range(len(key))]
    print("loss = {} || hidden_dim = {} || num_prop = {} || aggregate_type = {} || lr = {} || wd = {}".format(
          val_list[0], val_list[1], val_list[2], val_list[3], val_list[4], val_list[5] ))

    with open("config/msg_gnn.yaml", 'r') as ymlfile:
      cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

      for jj, kk in enumerate(key):
        if jj < 4:
          cfg['model'][kk] = val_list[jj]
        else:
          cfg['train'][kk] = val_list[jj]

    with open("config/msg_gnn_{:03d}.yaml".format(1+ii), 'w') as ymlfile:
      yaml.dump(cfg, ymlfile, explicit_start=True)

if __name__ == "__main__":
  test()