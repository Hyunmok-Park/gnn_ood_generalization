import yaml
import os
import argparse

def main(index):
    exp_list =[
        "/home/ubuntu/TorchGNN_project/GNN_exp/BP/TorchGNN_001_TorchGeoLoader_(10,9,8)_0.3_2021-Mar-10-05-18-44_train_64_10_add___"
    ]
    exp_dir = exp_list[0]
    split = [
        # 'exp2_test_16_0.3',
        # 'exp2_test_16_0.6',
        # 'exp2_test_16_0.75',
        # 'exp2_test_36_0.3',
        'exp2_test_100_0.3'
    ]
    for i in sorted(os.listdir(exp_dir)):
        # if i not in exclude:
        if "model_snapshot_best" in i:
                config_path = os.path.join(exp_dir, "config_test.yaml")
                cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

                cfg['dataset']['data_path'] = "data_temp"
                # cfg['dataset']['loader_name'] = "RandCRFData"
                # cfg['dataset']['name'] = "rand_crf"
                cfg['dataset']['split'] = split[index]
                cfg['runner'] = 'AlgorithmicInferenceRunner_bp'
                cfg['model']['name'] = 'BeliefPropagation'
                cfg['use_gpu'] = True
                cfg['exp_dir'] = "/home/ubuntu/TorchGNN_project/GNN_exp/BP/TorchGNN_001_TorchGeoLoader_(10,9,8)_0.3_2021-Mar-10-05-18-44_train_64_10_add___"
                cfg['model']['master_node'] = False
                cfg['model']['masking'] = False
                cfg['model']['jumping'] = False
                cfg['model']['interpol'] = False
                cfg['model']['skip_connection'] = False
                cfg['model']['SSL'] = False
                cfg['model']['train_pretext'] = False
                cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")


                with open('config/node_gnn_hyunmok_test{}.yaml'.format(index), 'w+') as ymlfile:
                    yaml.dump(cfg, ymlfile, explicit_start=True)

                os.system("python run_exp_local.py -c config/node_gnn_hyunmok_test{}.yaml -t".format(index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=1, help='index')
    args = parser.parse_args()

    main(args.index)