import collections
import numpy as np
import networkx as nx


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
  return node_list, adj.todense()


class NetworkTopology():

  def __init__(self, num_nodes, seed=1234):
    self.num_nodes = num_nodes
    self.seed = seed

  def generate(self, topology, p=None, argin1=None, argin2=None, argin3=None):
    if topology == 'star':
      G, W = self.star()
    elif topology == 'binarytree':
      G, W = self.binarytree()
    elif topology == 'path':
      G, W = self.path()
    elif topology == 'cycle':
      G, W = self.cycle()
    elif topology == 'wheel':
      G, W = self.wheel()
    elif topology == 'ladder':
      G, W = self.ladder()
    elif topology == 'circladder':
      G, W = self.circladder()
    elif topology == 'grid':
      G, W = self.grid()
    elif topology == 'barbell808':
      G, W = self.barbell(argin1, argin2)
    elif topology == 'barbell646':
      G, W = self.barbell2(argin1, argin2)
    elif topology == 'lollipop':
      G, W = self.lollipop()
    elif topology == 'bipartite':
      G, W = self.bipartite()
    elif topology == 'tripartite':
      G, W = self.tripartite()
    elif topology == 'complete':
      G, W = self.complete()
    elif topology == 'random':
      assert p is not None
      G, W = self.random_connected_np(p, seed=self.seed)
      self.seed += 1
    elif topology == 'samedegree':
      G, W = self.random_same_degree_with()
    elif topology == 'uniqdegree':
      G, W = self.random_unique_degree_of()
    elif topology == 'cylinder':
      G, W = self.cylinder(argin1, argin2)
    elif topology == 'torus':
      G, W = self.torus(argin1, argin2)
    elif topology == 'trilattice':
      G, W = self.trilattice(argin1, argin2)
    elif topology == 'trikite':
      G, W = self.trikite(argin1)
    elif topology == 'hexlattice':
      G, W = self.hexlattice(argin1, argin2)
    elif topology == 'nonalattice':
      G, W = self.nonalattice(argin1)
    elif topology == 'cube':
      G, W = self.cube(argin1, argin2, argin3)
    elif topology == 'ellcube':
      G, W = self.ellcube()
    elif topology == 'diamondcube':
      G, W = self.diamondcube(argin1, argin2)
    elif topology == 'crosslattice':
      G, W = self.crosslattice()
    elif topology == 'trigrid':
      G, W = self.trigrid()
    elif topology == 'trigridv2':
      G, W = self.trigridv2()
    else:
      raise Exception("network topology not supported: {}".format(topology))
    return G, W

  def graph_to_adjacency_matrix(self, G):
    n = G.number_of_nodes()
    W = np.zeros([n, n])
    for i in range(n):
      W[i, [j for j in G.neighbors(i)]] = 1
    return W

  def degree(self, G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt

  def mim_cycle_length(self, G):
      mcb = nx.minimum_cycle_basis(G)
      return min([len(cycle) for cycle in mcb])

  def unique_deg_preserved_seq(self, degree, count):
    num_node = sum(count)
    num_unique_deg = len(count)
    legitimate = False
    for i in range(100):
      cnt_new = np.random.randint(1, num_node - num_unique_deg + 1, size=num_unique_deg)
      if sum(cnt_new) == num_node and any(cnt_new != np.array(count)):
        legitimate = True
        break

    seq = []
    if legitimate:
      for i, c in enumerate(cnt_new):
        seq += [degree[i]] * c

    return seq

  # ----------- ACYCLIC GRAPH --------------
  def star(self):
    G = nx.star_graph(self.num_nodes - 1)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def binarytree(self):
    r = 2
    h = int(np.log2(self.num_nodes))
    G = nx.balanced_tree(r, h)
    for i in range(self.num_nodes, r**(1 + h) - 1):
      G.remove_node(i)

    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def path(self):
    G = nx.path_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  # ----------- CYCLIC GRAPH --------------
  def cycle(self):
    G = nx.cycle_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def wheel(self):
    G = nx.wheel_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def ladder(self):
    n = int(self.num_nodes / 2)
    G = nx.ladder_graph(n)
    if self.num_nodes > 2 * n:
      G.add_edge(2 * n - 1, 2 * n)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def circladder(self):
    n_cycle = int(self.num_nodes / 2)
    G = nx.circular_ladder_graph(n_cycle)
    if self.num_nodes > 2 * n_cycle:
      G.add_edge(2 * n_cycle - 1, 2 * n_cycle)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def grid(self):
    sqrtN = int(np.sqrt(self.num_nodes))
    assert self.num_nodes == sqrtN**2
    G = nx.grid_2d_graph(sqrtN, sqrtN)
    W = np.zeros([self.num_nodes, self.num_nodes])
    nodes = [n for n in G.nodes]
    for i in range(self.num_nodes):
      for j in G.neighbors(nodes[i]):
        W[i, sqrtN * j[0] + j[1]] = 1

    node_map = {gg: ii for ii, gg in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, node_map)
    return G, W

  def barbell(self, nc, np):
    if nc is None or np is None:
        nc = self.num_nodes // 2
        np = self.num_nodes % 2
    assert 2 * nc + np == self.num_nodes
    G = nx.barbell_graph(nc, np)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def barbell2(self, nc, np):
    if nc is None or np is None:
      if(self.num_nodes == 100):
        nc = 48
        np = 4
      elif(self.num_nodes == 64):
        nc = 30
        np = 4
      elif (self.num_nodes == 36):
        nc = 16
        np = 4
      elif (self.num_nodes == 16):
        nc = 6
        np = 4
      elif (self.num_nodes == 400):
        nc = 198
        np = 4
      elif (self.num_nodes == 900):
        nc = 448
        np = 4

        # nc = self.num_nodes // 2
        # np = self.num_nodes % 2
    assert 2 * nc + np == self.num_nodes
    G = nx.barbell_graph(nc, np)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def lollipop(self):
    m = int(np.ceil(self.num_nodes / 2)) + 1
    n = self.num_nodes - m
    G = nx.lollipop_graph(m, n)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def bipartite(self):
    n1 = self.num_nodes // 2
    n2 = self.num_nodes - n1
    G = nx.complete_bipartite_graph(n1, n2)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def tripartite(self):
    n1 = self.num_nodes // 3
    n2 = self.num_nodes // 3
    n3 = self.num_nodes - n1 - n2
    G = nx.complete_multipartite_graph(n1, n2, n3)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def complete(self):
    G = nx.complete_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  # ----------- RANDOM GRAPH --------------
  def random_connected_np(self, p, seed=1234):
    G = nx.fast_gnp_random_graph(self.num_nodes, p, seed=seed)
    cnt = 2
    while not nx.is_connected(G):
      G = nx.fast_gnp_random_graph(self.num_nodes, p, seed=seed * cnt)
      cnt += 1

    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def random_same_degree_with(self, G, num_graphs):
    W = self.graph_to_adjacency_matrix(G)
    seq_deg, _, _ = self.degree(G)
    G_list = [G]
    W_list = [W]

    cnt = 0
    for i in range(1, 1000):
      legitimate = True
      try:
        G_new = nx.random_degree_sequence_graph(seq_deg)
        for j in range(len(G_list)):
          if nx.is_isomorphic(G_list[j], G_new) or not nx.is_connected(G_new):
            legitimate = False
            break
        if legitimate:
          G_list.append(G_new)
          W_list.append(self.graph_to_adjacency_matrix(G_new))
          cnt += 1
      except:
        pass

      if cnt == num_graphs:
        break

    return G_list[1:], W_list[1:]

  def random_unique_degree_of(self, G, num_graphs):
    W = self.graph_to_adjacency_matrix(G)
    _, deg, count = self.degree(G)
    G_list = [G]
    W_list = [W]

    cnt = 0
    for i in range(1, 1000):
      legitimate = True
      try:
        seq = self.unique_deg_preserved_seq(deg, count)
        G_new = nx.random_degree_sequence_graph(seq)
        for j in range(len(G_list)):
          if nx.is_isomorphic(G_list[j], G_new) or not nx.is_connected(G_new):
            legitimate = False
            break
        if legitimate:
          G_list.append(G_new)
          W_list.append(self.graph_to_adjacency_matrix(G_new))
          cnt += 1
      except:
        pass

      if cnt == num_graphs:
        break

    return G_list[1:], W_list[1:]

  # ----------- SPECIAL GRAPH --------------
  def cylinder(self, nv_top, height):
    if nv_top is None or height is None:
      nv_top, height = 4,3
    nv = 2 * nv_top
    g1 = NetworkTopology(nv)
    G1, _ = g1.circladder()
    g2 = NetworkTopology(nv_top)
    for h in range(height - 1):
      G2, _ = g2.cycle()
      mapping = {i: nv + i for i in range(nv_top)}
      G2 = nx.relabel_nodes(G2, mapping)
      G1 = nx.compose(G1, G2)
      for i in range(nv_top):
        G1.add_edge(nv - nv_top + i, nv + i)
      nv = len(G1)

    W = self.graph_to_adjacency_matrix(G1)
    return G1, W

  def torus(self, nv_top, height):
    if nv_top is None or height is None:
      nv_top, height = 4, int((self.num_nodes/4)-1)
    G, _ = self.cylinder(nv_top, height)
    nv = len(G)
    for i in range(nv_top):
      G.add_edge(i, nv - nv_top + i)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def trilattice(self, nrow, ncol):
    if nrow is None or ncol is None:
      nrow, ncol = 1, self.num_nodes - 2
    G = nx.triangular_lattice_graph(nrow, ncol)
    node_map = {gg: ii for ii, gg in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, node_map)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def trikite(self, n):
    n = int((self.num_nodes-4)/3 + 1)
    G1, _ = self.trilattice(1, 2)
    for i in range(1, n):
      G2, _ = self.trilattice(1, 2)
      mapping = {j: 3 * i + j for j in range(4)}
      G2 = nx.relabel_nodes(G2, mapping)
      G1 = nx.compose(G1, G2)
    W = self.graph_to_adjacency_matrix(G1)

    return G1, W

  def hexlattice(self, nrow, ncol):
    if nrow is None or ncol is None:
      nrow, ncol = 2,2
    G = nx.hexagonal_lattice_graph(nrow, ncol)
    node_map = {gg: ii for ii, gg in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, node_map)
    W = self.graph_to_adjacency_matrix(G)

    return G, W

  def nonalattice(self, ncol):
    g1 = NetworkTopology(9)
    G1, _ = g1.cycle()
    g2 = NetworkTopology(9)
    nv = 9
    for i in range(ncol - 1):
      G2, _ = g2.cycle()
      mapping = {i: nv - 2 + i for i in range(9)}
      G2 = nx.relabel_nodes(G2, mapping)
      G1 = nx.compose(G1, G2)
      nv = len(G1)

    W = self.graph_to_adjacency_matrix(G1)
    return G1, W

  def cube(self, l, w, d):
    g = nx.grid_graph(dim=[l, w, d])
    node_map = {gg: ii for ii, gg in enumerate(g.nodes)}
    G = nx.relabel_nodes(g, node_map)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def ellcube(self):
    G1, _ = self.cylinder(4, 2)
    nv = len(G1)
    g2 = NetworkTopology(4)
    G2, _ = g2.cycle()
    mapping = {i: nv + i for i in range(4)}
    G2 = nx.relabel_nodes(G2, mapping)
    w = [6, 7, 11, 10]
    for i, v in enumerate(G2.nodes()):
      G1.add_edge(w[i], v)
    G1 = nx.compose(G1, G2)

    W = self.graph_to_adjacency_matrix(G1)
    return G1, W

  def diamondcube(self, n, height):
    nv = 4
    G1, _ = NetworkTopology(nv).cycle()
    for i in range(1, n):
      G2, _ = NetworkTopology(nv).cycle()
      mapping = {j: 0 if j == 0 else (nv - 1) * i + j for j in range(1, 4)}
      mapping[0] = 0
      G2 = nx.relabel_nodes(G2, mapping)
      G1 = nx.compose(G1, G2)

    W = self.graph_to_adjacency_matrix(G1)
    return G1, W

  def crosslattice(self):
    G1, _ = NetworkTopology(6).ladder()
    G2, _ = NetworkTopology(6).ladder()
    mapping = {0: 6, 2: 7, 3: 8, 5: 9}
    G2 = nx.relabel_nodes(G2, mapping)
    G1 = nx.compose(G1, G2)
    W = self.graph_to_adjacency_matrix(G1)
    return G1, W

  def trigrid(self):
    G, _ = self.grid()
    nv = len(G)
    n = int(np.sqrt(nv))
    for r in range(n - 1):
      for c in range(n - 1):
        G.add_edge(r * n + c, (r + 1) * n + c + 1)

    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def trigridv2(self):
    G, _ = self.grid()
    nv = len(G)
    n = int(np.sqrt(nv))
    for r in range(n - 1):
      for c in range(n - 1):
        G.add_edge(r * n + c, (r + 1) * n + c + 1)

    G.add_edge(2, 5)
    G.add_edge(5, 8)
    G.add_edge(7, 10)
    G.add_edge(10, 13)
    W = self.graph_to_adjacency_matrix(G)
    return G, W