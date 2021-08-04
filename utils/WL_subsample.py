from sklearn.decomposition import PCA
import numpy as np
import pickle
from tqdm import tqdm
import os
import shutil

p = pickle.load(open("data_temp/WS_flex_graph_100_bimodal_same_plane.p", "rb"))
prop_list, name_list = np.array(p['prop_list']), np.array(p['name_list'])

# data_full_mean = np.mean(prop_list, axis=0)
# data_full_scaled = prop_list - data_full_mean
# pca_full = PCA(n_components=6)
# pca_full.fit(data_full_scaled)
# X_low_full = pca_full.transform(data_full_scaled)

X_low_full = np.array(p['X_low'])

print("data_loaded")
c1, c2 = 0,0
c3, c4 = 0,0
for bin in [80]:
    graph_sample = np.zeros([bin, bin])
    count = 0
    H, x_axis, y_axis = np.histogram2d(X_low_full[:, 0], X_low_full[:, 1], bins=bin)
    # H, x_axis, y_axis = np.histogram2d(data_full_scaled[:, 0], data_full_scaled[:, 1], bins=bin)
    num_data = len(np.nonzero(H)[0])
    print(bin, "num_data", num_data)

    new_prop_list = []
    new_name_list = []
    new_X_low = []
    new_data_dict = {}
    for name, pca, prop in tqdm(zip(name_list, X_low_full, prop_list), desc='sub-sampling'):
    # for name, pca, prop in tqdm(zip(name_list, data_full_scaled, prop_list), desc='sub-sampling'):
        x = pca[0]
        y = pca[1]

        x_index = np.where(x_axis <= x)[-1][-1]
        y_index = np.where(y_axis <= y)[-1][-1]

        # name = name.split('/')[-1]

        if (x_index == bin):
            x_index -= 1
        if (y_index == bin):
            y_index -= 1

        if (graph_sample[x_index, y_index] == 0):
            graph_sample[x_index, y_index] = 1

            # name = name[0]
            new_name_list.append(name)
            new_prop_list.append(prop)
            new_X_low.append(pca)

            src = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal", name)
            src2 = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WL_flex_graphs_100_shell", name)
            try:
                os.mkdir("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_{}_SP".format(bin))
            except:
                pass
            dst = os.path.join("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_{}_SP".format(bin), name)

            try:
                shutil.copyfile(src, dst)
                c1 += 1
            except:
                c2 += 1

            try:
                shutil.copyfile(src2, dst)
                c3 += 1
            except:
                c4 += 1

            count += 1
            print("count: {}".format(count), end="\r", flush=True)
            if (len(np.nonzero(graph_sample)[0]) == num_data):
                print("break")
                break

        new_data_dict['prop_list'] = new_prop_list
        new_data_dict['name_list'] = new_name_list
        new_data_dict['X_low'] = new_X_low

    print("=======================================================")
    print(c1, c2)
    print(c3, c4)
    print("=======================================================")

with open("/home/ubuntu/TorchGNN_project/data_temp/WS_flex_graph_100_bimodal_pca_sub_{}_SP.p".format(bin), 'wb') as f:
    pickle.dump(new_data_dict, f)
    del new_data_dict




