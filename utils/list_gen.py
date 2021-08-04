import os
import pickle
import numpy as np

list_to_gen = ["/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_two_unimodal/both"]
for x, i in enumerate(list_to_gen):
    J_list = []
    name_list = []
    data_dict = {}
    number_of_graph = len(os.listdir(i))
    repeat_num = int(2000 / number_of_graph)
    number_to_supple = 2000 - repeat_num * number_of_graph
    #     print(number_of_graph, repeat_num, number_to_supple)

    for _ in range(repeat_num):
        for graph in os.listdir(i):
            data = pickle.load(open(os.path.join(i, graph), "rb"))
            J = data['J']
            J_list.append(J)
            name_list.append(graph)

    graph_list = os.listdir(i)
    if number_to_supple != 0:
        J_to_supple = np.random.choice(graph_list, number_to_supple, replace=False)
        for _ in J_to_supple:
            data = pickle.load(open(os.path.join(i, _), "rb"))
            J = data['J']
            J_list.append(J)
            name_list.append(_)

    print(len(J_list), len(name_list))
    data_dict['J_list'] = J_list
    data_dict['name_list'] = name_list

    with open('/home/ubuntu/TorchGNN_project/data_temp/group_two_unimodal_train_0.p',"wb") as f:
        pickle.dump(data_dict, f)

for x, i in enumerate(list_to_gen):
    J_list = []
    name_list = []
    data_dict = {}
    number_of_graph = len(os.listdir(i))
    repeat_num = int(100 / number_of_graph)
    number_to_supple = 100 - repeat_num * number_of_graph
    #     print(number_of_graph, repeat_num, number_to_supple)

    for _ in range(repeat_num):
        for graph in os.listdir(i):
            data = pickle.load(open(os.path.join(i, graph), "rb"))
            J = data['J']
            J_list.append(J)
            name_list.append(graph)

    graph_list = os.listdir(i)
    if number_to_supple != 0:
        J_to_supple = np.random.choice(graph_list, number_to_supple, replace=False)
        for _ in J_to_supple:
            data = pickle.load(open(os.path.join(i, _), "rb"))
            J = data['J']
            J_list.append(J)
            name_list.append(_)

    print(len(J_list), len(name_list))
    data_dict['J_list'] = J_list
    data_dict['name_list'] = name_list

    with open('/home/ubuntu/TorchGNN_project/data_temp/group_two_unimodal_val_0.p',"wb") as f:
        pickle.dump(data_dict, f)
