import os
import sys
import logging
import traceback
import numpy as np
from pprint import pprint
import torch

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config

torch.set_printoptions(profile='full')

def main(sample_id=1):
    args = parse_arguments()
    config = get_config(args.config_file, sample_id="{:03d}".format(sample_id))
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.use_gpu = config.use_gpu and torch.cuda.is_available()
    print("Using GPU", config.use_gpu)

    # log info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    # print(">" * 80)
    # pprint(config)
    # print("<" * 80)

    # Run the experiment
    try:
        runner = eval(config.runner)(config)
        if not args.test:
            runner.train

            import yaml
            cfg = yaml.load(open(config.save_dir + '/config.yaml', 'r'), Loader=yaml.FullLoader)
            cfg['exp_dir'] = config.save_dir # 날짜 형식
            cfg['dataset']['split'] = 'test'
            cfg['test']['test_model'] = config.save_dir + '/model_snapshot_best.pth'
            cfg['test']['batch_size'] = 1
            cfg_path = config.save_dir + '/config_test.yaml'
            with open(cfg_path, 'w') as ymlfile:
                yaml.dump(cfg, ymlfile, explicit_start=True)
            # os.system("python3 run_exp_local.py -c " + cfg_path + " -t")
        else:
            runner.test()
    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
    # SampleComputed().populate(reserve_jobs=True, suppress_errors=False)
