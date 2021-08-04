cimport numpy as np
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict

cimport cython

float_type = np.float32
ctypedef np.float_t float_type_t

int_type = np.int
ctypedef np.int_t int_type_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def build_grid_graph_BP(int height, int width, int kernel_size, int stride):
  """
    Build grid graphs with connected nodes within convolution kernel for mean-field
    
    Args:
      height: int
      width: int
      kernel_size: int, must be odd number 
      stride: int, must be odd number 

    Returns:
      pos: shape N X 2, numpy
      incidence_mat: shape N X E, scipy sparse coo
      msg_node: shape E X 2, numpy
      msg_adj: shape E X E, scipy sparse coo, transposed
  """
  cdef int_type_t H = height
  cdef int_type_t W = width
  cdef int_type_t K = kernel_size
  cdef int_type_t R = (kernel_size - 1) / 2
  cdef int_type_t S = stride
  cdef int_type_t R2 = R * R
  cdef int_type_t num_nodes = H * W
  cdef int_type_t num_edges = (K ** 2 - 1) * (H // S) * (W // S)
  cdef int_type_t ii, jj, xx, yy, kk, mm, nn, uu

  pos = np.zeros((num_nodes, 2), dtype=float_type)
  msg_node = np.zeros((num_edges, 2), dtype=int_type)  
  row_i = np.zeros(2*num_edges, dtype=int_type)
  col_i = np.zeros(2*num_edges, dtype=int_type)
  data_i = np.zeros(2*num_edges, dtype=float_type)
    
  idx_to_edge = {}
  node_to_in_edge = defaultdict(list) # map node ii to edge idx where edge is (xx, ii)
  node_to_out_edge = defaultdict(list) # map node ii to edge idx where edge is (ii, xx)
  
  # we loop over graph following row major order
  jj = 0
  uu = 0
  # build convolution kernel type graph
  for ii in range(num_nodes):
    xx = ii % W
    yy = ii / W
    pos[ii, 0] = yy
    pos[ii, 1] = xx

    if (xx % S) != 0 or (yy % S) != 0:
      continue

    for mm in range(-R, R+1):
      for nn in range(-R, R+1):
        if (mm == 0 and nn == 0) or (mm * mm + nn * nn) > R2:
          continue

        if xx + mm < 0 or xx + mm >= W or yy + nn < 0 or yy + nn >= H:
          continue

        kk = (xx + mm) + (yy + nn) * W

        # add edges        
        msg_node[jj, 0] = kk # node index
        msg_node[jj, 1] = ii # node index        
        idx_to_edge[jj] = (kk, ii)
        node_to_out_edge[kk] += [jj]
        node_to_in_edge[ii] += [jj]
        row_i[uu] = ii
        col_i[uu] = jj
        data_i[uu] = 1.0
        uu += 1
        jj += 1

  num_edges = jj
  msg_node = msg_node[:num_edges, :]

  # construct incidence matrix
  data_i = data_i[:uu]
  row_i = row_i[:uu]
  col_i = col_i[:uu]
  incidence_mat = coo_matrix((data_i, (row_i, col_i)), shape=(num_nodes, num_edges))

  # construct msg graph  
  jj = 0
  row_m = np.zeros(num_edges * (K**2), dtype=int_type)
  col_m = np.zeros(num_edges * (K**2), dtype=int_type)
  data_m = np.zeros(num_edges * (K**2), dtype=float_type)  
  for ii in range(num_nodes):
    for mm in node_to_out_edge[ii]:
      # loop over edges (ii, xx)
      for nn in node_to_in_edge[ii]:
        # loop over edges (yy, ii)
        if idx_to_edge[nn][0] == idx_to_edge[mm][1]:
          continue

        # N.B.: we are actually constructing msg_adj.transpose()
        row_m[jj] = mm
        col_m[jj] = nn
        data_m[jj] = 1.0
        jj += 1

  data_m = data_m[:jj]
  row_m = row_m[:jj]
  col_m = col_m[:jj]
  msg_adj = coo_matrix((data_m, (row_m, col_m)), shape=(num_edges, num_edges))  

  return pos, incidence_mat, msg_node, msg_adj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def build_grid_graph_mean_field(int height, int width, int kernel_size, int stride):
  """
    Build grid graphs with connected nodes within convolution kernel for mean-field
    
    Args:
      height: int
      width: int
      kernel_size: int, must be odd number 

    Returns:
      pos: shape N X 2, numpy      
      msg_node: shape E X 2, numpy      
  """
  cdef int_type_t H = height
  cdef int_type_t W = width
  cdef int_type_t K = kernel_size
  cdef int_type_t R = (kernel_size - 1) / 2
  cdef int_type_t R2 = R * R
  cdef int_type_t S = stride
  cdef int_type_t num_nodes = H * W
  cdef int_type_t num_edges = (K ** 2 - 1) * (H // S) * (W // S)
  cdef int_type_t ii, jj, xx, yy, kk, mm, nn

  msg_node = np.zeros((num_edges, 2), dtype=int_type)
  pos = np.zeros((num_nodes, 2), dtype=float_type)
  
  # we loop over graph following row major order
  jj = 0  
  # build convolution kernel type graph
  for ii in range(num_nodes):
    xx = ii % W
    yy = ii / W
    pos[ii, 0] = yy
    pos[ii, 1] = xx

    if (xx % S) != 0 or (yy % S) != 0:
      continue

    for mm in range(-R, R+1):
      for nn in range(-R, R+1):
        if (mm == 0 and nn == 0) or (mm * mm + nn * nn) > R2:
          continue

        if xx + mm < 0 or xx + mm >= W or yy + nn < 0 or yy + nn >= H:
          continue

        # add edges
        kk = (xx + mm) + (yy + nn) * W              
        msg_node[jj, 0] = kk # node index
        msg_node[jj, 1] = ii # node index       
        jj += 1

  num_edges = jj
  msg_node = msg_node[:num_edges, :]
  
  return pos, msg_node
