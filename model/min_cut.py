import numpy as np
import networkx as nx
from networkx.algorithms.flow import maximum_flow

__all__ = ['MinCut']

class MinCut(object):

    def __init__(self, G, J, b):
        super(MinCut, self).__init__()
        self.G = G
        self.J = J
        self.b = b

    def inference(self):
        DiG = self.G.to_directed()
        edge_array = np.array(DiG.edges)

        assert (self.J >= 0).all() # ferromagnetic

        for ii, edge in enumerate(edge_array):
            DiG.add_edge(edge[0], edge[1], capacity= self.J[int(edge[0]), int(edge[1])])

        DiG.add_node('s')
        DiG.add_node('t')

        for ii, bias in enumerate(self.b):
            if bias < 0:
                DiG.add_edge('s', ii, capacity=np.abs(bias))
            else:
                DiG.add_edge(ii, 't', capacity=np.abs(bias))

        _, partition = nx.minimum_cut(DiG, 's', 't')

        src = list(partition[0])
        src.remove('s')

        sink = list(partition[1])
        sink.remove('t')

        map_state = np.zeros((len(self.b), 1))
        map_state[src] = -1 # (or -1)
        map_state[sink] = 1

        return map_state
