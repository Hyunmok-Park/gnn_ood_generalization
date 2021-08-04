#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from math import exp
from math import *
import networkx as nx
import progressbar


def J_mat(Graph):
    n = len(Graph.nodes)
    J = np.zeros([n, n])
    for i in Graph.edges:
        J[i[0]][i[1]] = abs(round(random.gauss(0, 1), 5))
        J[i[1]][i[0]] = J[i[0]][i[1]]
    return J


def b_mat(node):
    B = np.zeros([node, 1])
    for i in range(node):
        b = round(random.gauss(0, 1), 5)
        B[i, 0] = b
    return B


def x_mat(node, option="decrete"):
    bar = progressbar.ProgressBar()
    if (option == "decrete"):
        x = np.zeros((node * (2 ** node)))
        l = 0
        for i in bar(range(2 ** node)):
            i = format(i, 'b').zfill(node)
            for k in i:
                if (k == '0'):
                    k = 0
                x[l] = k;
                l = l + 1;
        x = x.reshape(-1, node)

    if (option == "continuous"):
        x = np.zeros([2 ** node, node])
        for i in range(2 ** node):
            for l in range(node):
                x[i][l] = random.gauss(0, 1)
    return x


def Joint_P(x, b, J):
    node = int(len(b))
    P = np.zeros((2 ** node, 1))
    l = 0
    for i in x:
        p = exp(np.matmul(b.T, i) + 0.5 * np.matmul(np.matmul(i, J), i.T))
        P[l] = p
        l = l + 1
    S = np.sum(P)
    P = P / S
    return P


def Mar_P(x, JP):
    node = int(log2(int(len(x))))
    MarP = []
    for N in range(node):
        prob = 0
        for i in range(2 ** node):
            if (x[i, N] == 1):
                prob1 = JP[i]
                prob = prob1 + prob
        MarP.append(prob.tolist()[0])
    return MarP


def Belief_Prop(Graph, J, b, T):
    def SumProduct(Graph, message, node, option, exception=-1):
        neighbor = Graph.neighbors(node)
        result = 1
        if (option == "plus"):
            for i in neighbor:
                if (i == int(exception)):
                    result = result * 1
                else:
                    result = message[0, i, node] * result
        if (option == "minus"):
            for i in neighbor:
                if (i == int(exception)):
                    result = result * 1
                else:
                    result = message[1, i, node] * result
        return result

    n = len(Graph.nodes())
    message = np.ones([2, n, n])
    belief = np.ones([2, 1, n])
    for k in range(T):
        for i in range(n):
            for j in Graph.neighbors(i):
                message[0, j, i] = exp(J[i, j] + b[j]) * SumProduct(Graph, message, j, "plus", i) + exp(
                    -J[i, j] - b[j]) * SumProduct(Graph, message, j, "minus", i)
                message[1, j, i] = exp(-J[i, j] + b[j]) * SumProduct(Graph, message, j, "plus", i) + exp(
                    J[i, j] - b[j]) * SumProduct(Graph, message, j, "minus", i)

                Sum = message[0, j, i] + message[1, j, i]
                message[0, j, i] = message[0, j, i] / Sum
                message[1, j, i] = message[1, j, i] / Sum

            belief[0, 0, i] = exp(b[i]) * SumProduct(Graph, message, i, "plus")
            belief[1, 0, i] = exp(-b[i]) * SumProduct(Graph, message, i, "minus")

            Sum = belief[0, 0, i] + belief[1, 0, i]
            belief[0, 0, i] = belief[0, 0, i] / Sum
            belief[1, 0, i] = belief[1, 0, i] / Sum
    return belief


def Max_product(Graph, J, b, aa=1, bb=1):
    def SumProduct(Graph, message, node, option, exception=-1):
        neighbor = Graph.neighbors(node)
        result = 1
        if (option == "plus"):
            for i in neighbor:
                if (i == int(exception)):
                    result = result * 1
                else:
                    result = message[0, i, node] * result
        if (option == "minus"):
            for i in neighbor:
                if (i == int(exception)):
                    result = result * 1
                else:
                    result = message[1, i, node] * result
        return result

    n = len(Graph.nodes())
    message = np.ones([2, n, n])
    message0 = np.ones([2, n, n])
    message1 = np.ones([2, n, n])
    message_ = np.ones([2, n, n])

    belief = np.ones([2, 1, n])
    for k in range(20):
        for i in range(n):
            for j in Graph.neighbors(i):
                message[0, j, i] = max(aa * exp(J[i, j] + b[j]) * SumProduct(Graph, message0, j, "plus", i) + bb * exp(
                    J[i, j] + b[j]) * SumProduct(Graph, message1, j, "plus", i),
                                       aa * exp(-J[i, j] - b[j]) * SumProduct(Graph, message0, j, "minus", i) + bb * exp(
                                           -J[i, j] - b[j]) * SumProduct(Graph, message1, j, "minus", i))
                message[1, j, i] = max(aa * exp(-J[i, j] + b[j]) * SumProduct(Graph, message0, j, "plus", i) + bb * exp(
                    -J[i, j] + b[j]) * SumProduct(Graph, message1, j, "plus", i),
                                       aa * exp(J[i, j] - b[j]) * SumProduct(Graph, message0, j, "minus", i) + bb * exp(
                                           J[i, j] - b[j]) * SumProduct(Graph, message1, j, "minus", i))
                '''
                message[0,j,i] = max(exp(J[i,j] + b[j]) * SumProduct(Graph, message, j, "plus", i), exp(-J[i,j] - b[j]) * SumProduct(Graph, message, j, "minus", i))
                message[1,j,i] = max(exp(-J[i,j] + b[j]) * SumProduct(Graph, message, j, "plus", i) ,exp(J[i,j] - b[j]) * SumProduct(Graph, message, j, "minus", i))
                '''
                Sum = message[0, j, i] + message[1, j, i]
                message[0, j, i] = message[0, j, i] / Sum
                message[1, j, i] = message[1, j, i] / Sum

                message_ = message1
                message1 = message
                message0 = message_

            belief[0, 0, i] = exp(b[i]) * SumProduct(Graph, message, i, "plus")
            belief[1, 0, i] = exp(-b[i]) * SumProduct(Graph, message, i, "minus")

            Sum = belief[0, 0, i] + belief[1, 0, i]
            belief[0, 0, i] = belief[0, 0, i] / Sum
            belief[1, 0, i] = belief[1, 0, i] / Sum

    return belief


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))

        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def SIGMA(Graph, Mat, node, exception=-1):
    result = 0
    for k in Graph.neighbors(node):
        if (k == int(exception)):
            result = result
        else:
            result = result + Mat[k, node]
    return result


def GaBP(Graph, A_mat, b_mat):
    b = b_mat
    n = len(Graph.nodes)
    P_mat1 = np.zeros([n, n])
    P_mat2 = np.zeros(n)
    mean_mat1 = np.ones([n, n])
    mean_mat2 = np.zeros(n)
    for i in range(n):
        P_mat1[i, i] = A_mat[i, i]
        mean_mat1[i, i] = b[i] / A_mat[i, i]

    before = np.zeros([2, n, n])
    after = np.zeros([2, n, n])
    value = True

    while (value):
        for i in range(n):
            for j in range(n):
                before[0, i, j] = P_mat1[i, j]
                before[1, i, j] = mean_mat1[i, j]

        for i in range(n):
            P_mat2[i] = P_mat1[i, i] + SIGMA(Graph, P_mat1, i)
            mean_mat2[i] = (P_mat1[i, i] * mean_mat1[i, i] + SIGMA(Graph, P_mat1 * mean_mat1, i)) * (1 / P_mat2[i])

            for j in Graph.neighbors(i):
                P_mat1[i, j] = -(A_mat[i, j] ** 2) / (P_mat2[i] - P_mat1[j, i])
                mean_mat1[i, j] = (P_mat2[i] * mean_mat2[i] - P_mat1[j, i] * mean_mat1[j, i]) / A_mat[i, j]

        for i in range(n):
            for j in range(n):
                after[0, i, j] = P_mat1[i, j]
                after[1, i, j] = mean_mat1[i, j]

        if (np.all(abs(before - after) < 0.001)):
            value = False

    mean_mat3 = np.zeros(n)
    P_mat3 = np.zeros(n)

    for i in range(n):
        for k in Graph.neighbors(i):
            mean_mat3[i] = (P_mat1[i, i] * mean_mat1[i, i] + SIGMA(Graph, P_mat1 * mean_mat1, i)) / (
                    P_mat1[i, i] + SIGMA(Graph, P_mat1, i))
            P_mat3[i] = P_mat1[i, i] + SIGMA(Graph, P_mat1, i)

    return mean_mat3


def Energy_min_cut(Graph, J, b):
    G = Graph.copy().to_directed()
    n = len(G.nodes())

    for i in range(n):
        for j in range(n):
            if (J[i, j] != 0):
                G.add_edge(i, j, capacity= J[i, j])

    G.add_node(n)
    G.add_node(n+1)
    for i in range(len(b)):
        if (b[i] < 0):
            G.add_edge(n, i, capacity=abs(b[i]))
        else:
            G.add_edge(i, n+1, capacity=abs(b[i]))

    graph = np.zeros([2, n + 2, n + 2])
    for i in list(G.edges(data=True)):
        graph[0, i[0], i[1]] = i[2]['capacity']

    graph_, re_graph_ = max_flow(n, n + 1, G, graph)
    cut_list = []
    for i in range(n + 2):
        for j in range(n + 2):
            if (graph_[1, i, j] > 0 and re_graph_[i, j] == 0):
                cut_list.append((i, j))
    return cut_list


def max_flow(source, sink, Graph, graph):
    paths = nx.all_simple_paths(Graph, source, sink)
    for path in paths:
        re_graph = graph[0] - graph[1]
        capacity_list = []
        for i in range(len(path) - 1):
            capacity_list.append(re_graph[path[i], path[i + 1]])
        max_flow = min(capacity_list)
        for i in range(len(path) - 1):
            graph[1, path[i], path[i + 1]] = graph[1, path[i], path[i + 1]] + max_flow
            graph[1, path[i + 1], path[i]] = graph[1, path[i + 1], path[i]] - max_flow
    return graph, graph[0] - graph[1]


def sorting(Graph, cut, b):
    G = Graph.copy()
    n = len(G.nodes())

    G.add_node(n)
    G.add_node(n + 1)

    for i in range(len(b)):
        if (b[i] < 0):
            G.add_edge(n, i)
        else:
            G.add_edge(i, n + 1)

    for i in cut:
        G.remove_edge(i[0], i[1])

    source_list = []
    sink_list = []
    for i in G.nodes():
        if (i != n and i != n + 1):
            if (len(list(nx.all_simple_paths(G, i, n))) == 0):
                sink_list.append(i)
            else:
                source_list.append(i)
    return source_list, sink_list
