import os
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shutil
import argparse
import pickle
import numpy as np

data = pickle.load(open("data_temp/WL_flex_graphs_100_shell_UNION.p", "rb"))['prop_list']
name_list = pickle.load(open("data_temp/WL_flex_graphs_100_shell_UNION.p", "rb"))['name_list']
print("data_loaded")
data = np.array(data)

data_mean = np.mean(data, axis=0)
data_scaled = (data - data_mean)
pca = PCA(n_components=6)
pca.fit(data_scaled)
X_low = pca.transform(data_scaled)
cmap = plt.cm.bwr

# index = []
# for x, i in enumerate(data):
#     if 48<i[0]<52:
#         index.append(x)

x6 = []
for x, i in enumerate(X_low):
    if -41 < i[0] < -39 and -20.5 < -i[1] < -19.5:
        x6.append(x)

print(len(x6))

x7 = []
for x, i in enumerate(X_low):
    if 49 < i[0] < 51 and -21 < -i[1] < -19:
        x7.append(x)

print(len(x7))

for i in tqdm(name_list[x6]):
    graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell_UNION", i)
    dest = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_two_unimodal/lowmean", i)
    shutil.copyfile(graph_path, dest)
    # except:
    #     graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell", i)
    #     dest = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_3.5", i)
    #     shutil.copyfile(graph_path, dest)

for i in tqdm(name_list[x7]):
    graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell_UNION", i)
    dest = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_two_unimodal/highmean", i)
    shutil.copyfile(graph_path, dest)
    # except:
    #     graph_path = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell", i)
    #     dest = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/unique_graph_100/group_3.5", i)
    #     shutil.copyfile(graph_path, dest)
