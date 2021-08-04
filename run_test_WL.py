import argparse
import yaml
import os

def main(index):
    path = "/home/ubuntu/TorchGNN_project/GNN_exp/V=16/train_0.75"
    exp_list = [os.path.join(path, i) for i in sorted(os.listdir(path)) if "MsgGNN" not in i and "DS" not in i]
    print(len(exp_list))

    print(index)
    exp_list = [
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-07-54_train_100_group_1_64_10_add__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-07-57_train_100_group_1_64_10_add__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-00_train_100_group_1_64_10_att__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-03_train_100_group_1_64_10_att__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-06_train_100_group_2_64_10_add__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-09_train_100_group_2_64_10_add__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-12_train_100_group_2_64_10_att__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-15_train_100_group_2_64_10_att__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-18_train_100_group_3_64_10_add__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-21_train_100_group_3_64_10_add__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-24_train_100_group_3_64_10_att__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-27_train_100_group_3_64_10_att__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-30_train_100_group_0_64_10_add__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-33_train_100_group_0_64_10_add__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-36_train_100_group_0_64_10_att__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-39_train_100_group_0_64_10_att__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-42_train_100_group_4_64_10_add__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-45_train_100_group_4_64_10_add__SK_IP",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-48_train_100_group_4_64_10_att__SK_",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_TorchGeoLoader_2021-Feb-20-18-08-51_train_100_group_4_64_10_att__SK_IP",
"/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-40_train_100_group_0_64_10_add",
"/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-43_train_100_group_1_64_10_add",
"/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-47_train_100_group_2_64_10_add",
"/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-49_train_100_group_3_64_10_add",
"/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-07-16-37-53_train_100_group_4_64_10_add",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-28_train_100_group_4_64_10_att___",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-31_train_100_group_3_64_10_att___",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-34_train_100_group_2_64_10_att___",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-12-20-13-37_train_100_group_1_64_10_att___",
# "/home/ubuntu/TorchGNN_project/GNN_exp/V=100/TorchGNN_001_Torchloader_2021-Jan-18-18-56-21_train_100_group_0_64_10_att___"
    ]
    exp_dir = exp_list[index]
    for i in sorted(os.listdir(exp_dir)):
        # if i not in exclude:
            if "model_snapshot_best" in i:
                print(i)
                # config_path = os.path.join(exp_dir, "config_test.yaml")
                # cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
                #
                # cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['split'] = "exp2_test_16_0.3"
                # cfg['exp_dir'] = exp_dir
                # # cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
                # # cfg['model']['name'] = "TorchGNN_MsgGNN"
                # # cfg['dataset']['data_path'] = "data_temp"
                # # cfg['dataset']['split'] = "exp2_test_16"
                # cfg['test']['test_model'] = os.path.join(exp_dir, i)
                # cfg['model']['master_node'] = False
                # cfg['model']['masking'] = False
                #
                # with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                #     yaml.dump(cfg, ymlfile, explicit_start=True)

                # os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))
    ###
                # config_path = os.path.join(exp_dir, "config_test.yaml")
                # cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
                #
                # cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['split'] = "exp2_test_36_0.3"
                # cfg['exp_dir'] = exp_dir
                # # cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
                # # cfg['model']['name'] = "TorchGNN_MsgGNN"
                # # cfg['dataset']['data_path'] = "data_temp"
                # # cfg['dataset']['split'] = "exp2_test_16"
                # cfg['test']['test_model'] = os.path.join(exp_dir, i)
                # cfg['model']['master_node'] = False
                # cfg['model']['masking'] = False
                #
                # with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                #     yaml.dump(cfg, ymlfile, explicit_start=True)
                #
                # os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))

    ###
                config_path = os.path.join(exp_dir, "config.yaml")
                cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

                cfg['dataset']['data_path'] = "data_temp"
                cfg['dataset']['split'] = "WS_flex_graph_100_bimodal_pca_sub_80_SP_JB_2"
                cfg['exp_dir'] = exp_dir
                cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
                # cfg['model']['name'] = "TorchGNN_MsgGNN"
                # cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['split'] = "exp2_test_16"
                cfg['test']['test_model'] = os.path.join(exp_dir, i)
                cfg['model']['master_node'] = False
                cfg['model']['masking'] = False
                cfg['model']['jumping'] = False
                cfg['model']['interpol'] = False
                cfg['model']['skip_connection'] = False
                cfg['model']['SSL'] = False
                cfg['model']['train_pretext'] = False


                with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                    yaml.dump(cfg, ymlfile, explicit_start=True)

                os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))


    ###
                # config_path = os.path.join(exp_dir, "config_test.yaml")
                # cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
                #
                # cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['split'] = "exp2_test_16_0.6"
                # cfg['exp_dir'] = exp_dir
                # # cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
                # # cfg['model']['name'] = "TorchGNN_MsgGNN"
                # # cfg['dataset']['data_path'] = "data_temp"
                # # cfg['dataset']['split'] = "exp2_test_16"
                # cfg['test']['test_model'] = os.path.join(exp_dir, i)
                # cfg['model']['master_node'] = False
                # cfg['model']['masking'] = False
                #
                # with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                #     yaml.dump(cfg, ymlfile, explicit_start=True)
                #
                # os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))

    ###
                # config_path = os.path.join(exp_dir, "config_test.yaml")
                # cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
                #
                # cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['split'] = "exp2_test_16_0.75"
                # cfg['exp_dir'] = exp_dir
                # # cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
                # # cfg['model']['name'] = "TorchGNN_MsgGNN"
                # # cfg['dataset']['data_path'] = "data_temp"
                # # cfg['dataset']['split'] = "exp2_test_16"
                # cfg['test']['test_model'] = os.path.join(exp_dir, i)
                # cfg['model']['master_node'] = False
                # cfg['model']['masking'] = False
                #
                # with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                #     yaml.dump(cfg, ymlfile, explicit_start=True)
                #
                # os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))

                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=1, help='index')
    args = parser.parse_args()

    main(args.index)
