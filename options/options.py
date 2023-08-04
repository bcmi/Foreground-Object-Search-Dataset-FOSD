import argparse
import os
import yaml

def refresh_yaml_params(args, config_file):
    with open(config_file, 'r') as yaml_file:
        yaml_params = yaml.full_load(yaml_file.read())
    for arg in vars(args):
        value = getattr(args, arg)
        if arg in yaml_params and value != None:
            yaml_params[arg] = value
    refreshed_file = config_file.replace('.yaml', '_tmp.yaml')
    with open(refreshed_file, 'w') as f:
        yaml.dump(yaml_params, f)
    return refreshed_file

def parse_args():
    parser = argparse.ArgumentParser(description="Training Foreground Object Search Model ...")
    parser.add_argument('--gpu',  type=int, dest='gpu_id', help='gpu_id')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--cfg', type=str, dest='config_file', help='config file',
                        default="config_sfosd.yaml", choices=["config_sfosd.yaml", "config_rfosd.yaml"])
    # settings for network architecture
    parser.add_argument('--distill', type=str, dest='distill_type',
                        choices=['none', 'roiconcat', 'roicropresize', 'gloablconcat', \
                                 'roivectorconcat', 'gapconcat', 'onlyclassify'])
    parser.add_argument('--feature', type=str, dest='encoder_feature',
                        choices=['roi', 'gap'])
    parser.add_argument('--classify', type=bool, dest="encoder_classify", default=False)
    parser.add_argument('--backbone', type=str,
                        help='the architecture of backbone network')
    # settings for training
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--margin', type=float, dest="triplet_margin", default=None)
    parser.add_argument('--distill_weight', type=float, default=None)
    parser.add_argument('--classify_weight', type=float, default=None)
    parser.add_argument('--batch', type=int, dest="batch_size", default=None)
    parser.add_argument('--epoch', type=int, dest="max_epoch",  default=None)
    parser.add_argument('--lr', type=float, dest="init_lr", default=None,
                        help='initial learning rate for adam')
    # settings for evaluation
    parser.add_argument('--testset2', dest="eval_testset2",
                        type=bool, default=None)
    parser.add_argument('--testset1', dest="eval_testset1",
                        type=bool, default=None)
    parser.add_argument('--test_freq', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int,
                        help='number of dataloader workers')
    parser.add_argument('--save_freq', type=int)
    args = parser.parse_args()
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(proj_dir, 'config')
    # update default arguments to arguments given by command line
    config_file = os.path.join(config_dir, args.config_file)
    assert os.path.exists(config_file)
    new_config_file = refresh_yaml_params(args, config_file)
    return new_config_file, args.local_rank, args.config_file