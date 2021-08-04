import yaml
import os

for exp_dir in [
"GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-35_train_16_binarytree_64_10_add___",
# "GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-35_train_16_grid_64_10_add___",
# "GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-36_train_16_trigrid_64_10_add___",
# "GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-39_train_16_bipartite_64_10_add___",
# "GNN_exp/exp1/J0.3_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-42_train_16_all_64_10_add___"
#
#"GNN_exp/exp1/J0.6_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-45_train_16_binarytree_64_10_add___",
#"GNN_exp/exp1/J0.6_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-48_train_16_grid_64_10_add___",
#"GNN_exp/exp1/J0.6_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-51_train_16_trigrid_64_10_add___",
#"GNN_exp/exp1/J0.6_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-54_train_16_bipartite_64_10_add___",
#"GNN_exp/exp1/J0.6_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-14-49-57_train_16_all_64_10_add___"
#
# "GNN_exp/exp1/J0.9_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-20-57-26_train_16_binarytree_64_10_add___",
# "GNN_exp/exp1/J0.9_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-20-57-29_train_16_grid_64_10_add___",
# "GNN_exp/exp1/J0.9_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-20-57-32_train_16_trigrid_64_10_add___",
# "GNN_exp/exp1/J0.9_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-20-57-35_train_16_bipartite_64_10_add___"
# "GNN_exp/exp1/J0.9_b0.25_geo/TorchGNN_MsgGNN_001_TorchGeoLoader_2021-Feb-07-20-57-38_train_16_all_64_10_add___"
                ]:
    for nn in [16, 36, 64, 100]:
        for tt in ['path', 'binarytree', 'cycle', 'grid', 'circladder', 'ladder', 'cylinder', 'torus', 'trikite' ,'trilattice', 'trigrid', 'barbell2', 'barbell', 'bipartite', 'complete']:

            config_path = os.path.join(exp_dir, "config_test.yaml")
            cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

            cfg['dataset']['data_path'] = "data_temp/exp1_test/test_0.3/{}".format(nn)
            cfg['dataset']['split'] = "{}_0.3".format(tt)
            cfg['exp_dir'] = exp_dir
            cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")

            with open('config/node_gnn_hyunmok_test2.yaml', 'w') as ymlfile:
                yaml.dump(cfg, ymlfile, explicit_start=True)

            os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test2.yaml -t")


