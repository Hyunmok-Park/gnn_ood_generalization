import yaml
import os

for exp_dir in [
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-35_train_16_binarytree_64_10_add___",
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-35_train_16_grid_64_10_add___",
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-36_train_16_trigrid_64_10_add___",
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-39_train_16_bipartite_64_10_add___",
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-42_train_16_all_64_10_add___"
#
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.6_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-15-40-10_train_16_binarytree_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.6_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-15-40-11_train_16_grid_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.6_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-16-18-21_train_16_trigrid_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.6_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-16-18-23_train_16_bipartite_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.6_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-16-18-25_train_16_all_64_10_add"

# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.9_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-19-27-08_train_16_binarytree_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.9_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-19-27-11_train_16_grid_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.9_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-19-27-14_train_16_trigrid_64_10_add",
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.9_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-19-27-17_train_16_bipartite_64_10_add"
# "/home/ubuntu/pycharm_project_home/GNN_exp2/J0.9_b0.25_geo/TorchGNN_001_Torchloader_2020-Dec-23-19-27-20_train_16_all_64_10_add"
                ]:
    for nn in [16, 36, 64, 100]:
        for tt in ['path', 'binarytree', 'cycle', 'grid', 'circladder', 'ladder', 'cylinder', 'torus', 'trikite' ,'trilattice', 'trigrid', 'barbell2', 'barbell', 'bipartite', 'complete']:

            config_path = os.path.join(exp_dir, "config_test.yaml")
            cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

            cfg['dataset'][
                'data_path'] = "data_temp/exp1_test/test_0.3/{}".format(nn)
            cfg['dataset']['split'] = "{}_0.3".format(tt)
            cfg['runner'] = 'AlgorithmicInferenceRunner'
            cfg['model']['name'] = 'TreeReWeightedMessagePassing'
            cfg['model']['max_iter'] = 10
            cfg['model']['damping'] = 0.0
            cfg['model']['num_trees'] = 10
            cfg['use_gpu'] = False
            cfg['exp_dir'] = exp_dir
            cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")

            with open('config/node_gnn_hyunmok_test5.yaml', 'w') as ymlfile:
                yaml.dump(cfg, ymlfile, explicit_start=True)

            os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test5.yaml -t")

