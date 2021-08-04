import os
import pickle
import networkx as nx
import numpy as np

path = [
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_100_0.3",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_100_MSG_0.3",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_0.3",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_0.6",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_0.75",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_MSG_0.3",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_MSG_0.6",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_16_MSG_0.75",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_36_0.3",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_test_36_MSG_0.3",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_0/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_1/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_2/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4/train",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_0/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_1/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_2/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_100_0.3/group_4/val",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(3,2,1)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(4,3,2)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(5,4,3)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(6,5,4)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(7,6,5)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(8,7,6)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(9,8,7)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(10,9,8)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(11,10,9)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(12,11,10)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(13,12,11)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(14,13,12)_0.3/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(15,14,13)_0.3/train",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(3,2,1)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(4,3,2)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(5,4,3)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(6,5,4)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(7,6,5)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(8,7,6)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(9,8,7)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(10,9,8)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(11,10,9)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(12,11,10)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(13,12,11)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(14,13,12)_0.3/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.3/(15,14,13)_0.3/val",


"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(3,2,1)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(4,3,2)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(5,4,3)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(6,5,4)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(7,6,5)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(8,7,6)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(9,8,7)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(10,9,8)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(11,10,9)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(12,11,10)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(13,12,11)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(14,13,12)_0.6/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(15,14,13)_0.6/train",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(3,2,1)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(4,3,2)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(5,4,3)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(6,5,4)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(7,6,5)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(8,7,6)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(9,8,7)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(10,9,8)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(11,10,9)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(12,11,10)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(13,12,11)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(14,13,12)_0.6/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.6/(15,14,13)_0.6/val",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(3,2,1)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(4,3,2)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(5,4,3)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(6,5,4)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(7,6,5)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(8,7,6)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(9,8,7)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(10,9,8)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(11,10,9)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(12,11,10)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(13,12,11)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(14,13,12)_0.75/train",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(15,14,13)_0.75/train",

"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(3,2,1)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(4,3,2)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(5,4,3)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(6,5,4)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(7,6,5)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(8,7,6)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(9,8,7)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(10,9,8)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(11,10,9)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(12,11,10)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(13,12,11)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(14,13,12)_0.75/val",
"/home/ubuntu/TorchGNN_project/data_temp/exp2_train_16_0.75/(15,14,13)_0.75/val",
]


for i in path:
    first_graph = os.listdir(i)[0]
    graph = pickle.load(open(os.path.join(i, first_graph), "rb"))
    try:
        stdJ = graph['J_std']
    except:
        stdJ = graph['stdJ']

    try:
        stdb = graph['b_std']
    except:
        stdb = graph['stdb']


    print(i, stdJ, stdb)