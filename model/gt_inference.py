import numpy as np
from tqdm import tqdm


class Enumerate(object):

  def __init__(self, A, J, b):
    """
      Belief Propagation for Binary Markov Models

      A: shape N X N, binary adjacency matrix
      J: shape N X N, parameters of pairwise term
      b: shape N X 1, parameters of unary term
    """
    super(Enumerate, self).__init__()
    self.A = A
    self.J = np.multiply(J, A)
    self.b = b
    self.num_nodes = A.shape[0]
    self.num_states = 2
    self.states = [1, -1]

  def inference(self):
    node_states = [np.array(self.states) for _ in range(self.num_nodes)]
    grid_list = np.meshgrid(*node_states)
    grid_list = [nn.flatten() for nn in grid_list]
    prob = np.zeros((self.num_nodes, self.num_states))

    for ii in tqdm(range(self.num_nodes), desc='inference', leave=False):
      for jj, ss in enumerate(self.states):
        idx = np.where(grid_list[ii] == ss)[0]

        for kk in range(len(idx)):
          x = np.array([ll[idx[kk]] for ll in grid_list]).reshape(1, -1)  # shape 1 X N
          prob[ii][jj] += np.exp(x.dot(self.b) + 0.5 * x.dot(self.J).dot(x.transpose()))
  
    prob /= np.sum(prob, axis=1, keepdims=True)
    return prob

  def inference_map(self):
    node_states = [np.array(self.states) for _ in range(self.num_nodes)]
    grid_list = np.meshgrid(*node_states)
    grid_list = [nn.flatten() for nn in grid_list]
    grid_array = np.stack(grid_list).T
    prob = np.zeros(grid_array.shape[0])

    for ii, x in tqdm(enumerate(grid_array), desc='inference_map', leave=False):
      prob[ii] = np.exp(x.dot(self.b) + 0.5 * x.dot(self.J).dot(x.transpose()))

    prob /= prob.sum()
    map_state = grid_array[np.argmax(prob)].reshape(-1,1)
    return map_state


if __name__ == '__main__':
  A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
  J = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, -0.8, 0.9]])
  J = (J + J.transpose()) / 2.0
  #J = np.ones((3, 3)) * 0.5
  b = np.ones((3, 1)) * 0.5

  model = Enumerate(A, J, b)
  P = model.inference()
  print(P)
  