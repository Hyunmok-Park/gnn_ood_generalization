import os
import pickle
import networkx as nx
import collections
from sklearn.decomposition import PCA
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt


data = pickle.load(open("../data_temp/16_list_shell_noJ.p", "rb"))
name_list = data['name_list']
data = data['prop_list']
print("data_loaded")

data_mean = np.mean(data, axis=0)
data_scaled = data-data_mean
pca = PCA(n_components=6)
pca.fit(data_scaled)
X_low = pca.transform(data_scaled)

b = []
plt.scatter(-X_low[:, 0], -X_low[:, 1], s=1)
count1, count2 = 0,0
for x, (i, j) in tqdm(enumerate(zip(X_low[:, 0], X_low[:, 1]))):
    if -0.5 < -i < 0.5 and -1.9 < -j < -1.6:
        # plt.scatter(-i, -j, color='orange', s=2, marker='X')
        count1+=1
        # b.append(name_list[x])

    if -1 < -i < 0.25 and -4.9 < -j < -4.6:
        # plt.scatter(-i, -j, color='orange', s=2, marker='X')
        count2 += 1
        # b.append(name_list[x])
#         print("x")
print(count1, count2)
# plt.xlim(-1, 1)
# plt.ylim(-2.3, -1.5)
# plt.grid()
# print(len(b))
# plt.savefig("result.pdf")






