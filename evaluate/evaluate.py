import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import csv
from evaluate.base import get_model_score_on_testset2, eval_fine_ranking_on_testset1, eval_score_shape_ranking_on_testset2
from dataset.datasets import ClassName
from network.networks import StudentModel
from config.config import Config
import copy
from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

logger = None

class Evaluater(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics1 = cfg.test1_metrics
        self.metrics2 = cfg.test2_metrics

        self.test_results1 = dict()
        for cls in ClassName[:cfg.num_classes]+['overall']:
            self.test_results1[cls] = {k: [] for k in self.metrics1}
            
        self.test_results2 = dict()
        for cls in ClassName[:cfg.num_classes]+['overall']:
            self.test_results2[cls] = {k: [] for k in self.metrics2}

        self.test_method  = []
    
    def write2csv(self, test_results, suffix=''):
        csv_save_folder = os.path.join(PROJ_dir, "eval_results")
        if not os.path.exists(csv_save_folder):
            os.mkdir(csv_save_folder)
        csv_path = os.path.join(csv_save_folder, '{}{}.csv'.format(time.strftime('%Y%m%d_%H%M%S'), suffix))
        if suffix == '-testset1':
            metrics  = copy.deepcopy(self.metrics1)
        elif suffix == '-testset2':
            metrics  = copy.deepcopy(self.metrics2)
        metrics_tmp = metrics.copy()
        results  = test_results['overall']
        for cls in test_results.keys():
            if cls == 'overall':
                continue
            for k in metrics_tmp:
                tmp_k = f'{cls}-{k}'
                try:
                    results[tmp_k] = test_results[cls][k]
                except:
                    logger.info([cls, k, list(test_results.keys()), list(test_results[cls].keys())])
                metrics.append(tmp_k)
        header  = ['method'] + metrics
        method = self.test_method
        results['method'] = method
        rows = [header]
        for i in range(len(method)):
            row = [results[m][i] for m in header]
            rows.append(row)

        for name in header:
            if name not in metrics_tmp:
                continue
            cur_result = results[name]
            best_index = cur_result.index(max(cur_result))
            title = 'best {}'.format(name)
            row = [results[k][best_index] for k in header]
            row[0] = title
            rows.append(row)

        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ' + csv_path)

    def eval_afterTraining(self, model, threshold, test_on_set1 = True, test_on_set2 = True,
                           save_score = True, save_score_folder = "model_scores",
                           save_top20 = True, save_top20_folder = "top20"):
    
        if test_on_set1:
            self.test_method.append("Ours_S-FOSD_model")
            test_results1 = eval_fine_ranking_on_testset1(model, cfg)
            for k,v in test_results1.items():
                if isinstance(v, dict):
                    for k1,v1 in v.items():
                        self.test_results1[k][k1].append(v1)
                else:
                    self.test_results1[k].append(v)
            self.write2csv(self.test_results1,   '-testset1')

        if test_on_set2:      
            self.test_method.append("Ours_R-FOSD_model")      
            save_info = get_model_score_on_testset2(model, cfg, save_score, save_score_folder)
            test_results2 = eval_score_shape_ranking_on_testset2(cfg, threshold, save_info, save_top20, save_top20_folder)
            for k,v in test_results2.items():
                if isinstance(v, dict):
                    for k1,v1 in v.items():
                        self.test_results2[k][k1].append(v1)
                else:
                    self.test_results2[k].append(v)
            self.write2csv(self.test_results2,  '-testset2')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=1.2, help="Threshold of aspect ratio.")
    parser.add_argument('--testOnSet1', action='store_true', help="Test on testset1 or not.")
    parser.add_argument('--testOnSet2', action='store_true', help="Test on testset2 or not.")
    parser.add_argument('--saveScores', action='store_true', help="Whether to save the outputs(scores) of the model on testset2.")
    parser.add_argument('--saveTop20', action='store_true', help="Whether to save the top 20 search results on testset2.")
    parser.add_argument('--saveScorePath', type=str, default="model_scores", help="Path to save scores.")
    parser.add_argument('--saveTop20Path', type=str, default="top20", help="Path to save top 20 results.")
    args = parser.parse_args()
    
    ckpt_dir = os.path.join(PROJ_dir, 'checkpoints')
    assert os.path.exists(ckpt_dir), ckpt_dir
    
    weight_list = ["sfosd.pth", "rfosd.pth"]
    if args.testOnSet1 == True:
        config_file = os.path.join(PROJ_dir, 'config/config_sfosd.yaml')
        cfg = Config(config_file)
        device = torch.device('cuda:0')
        model = StudentModel(cfg).to(device).eval()
        weight_epoch = weight_list[0]
        evaluater = Evaluater(cfg)
        weight_file = os.path.join(ckpt_dir, weight_epoch)
        assert os.path.exists(weight_file), weight_file
        print('load weights ', weight_file)
        weights = torch.load(weight_file)
        model.load_state_dict(weights, strict=True)
        evaluater.eval_afterTraining(model, args.threshold, True, False)
    if args.testOnSet2 == True:
        config_file = os.path.join(PROJ_dir, 'config/config_rfosd.yaml')
        cfg = Config(config_file)
        device = torch.device('cuda:0')
        model = StudentModel(cfg).to(device).eval()
        weight_epoch = weight_list[1]
        evaluater = Evaluater(cfg)
        weight_file = os.path.join(ckpt_dir, weight_epoch)
        assert os.path.exists(weight_file), weight_file
        print('load weights ', weight_file)
        weights = torch.load(weight_file)
        model.load_state_dict(weights, strict=True)
        evaluater.eval_afterTraining(model, args.threshold, False, True, args.saveScores, args.saveScorePath, args.saveTop20, args.saveTop20Path)