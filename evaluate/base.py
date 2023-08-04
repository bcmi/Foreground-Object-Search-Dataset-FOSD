import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataset.datasets import TestDataset, ClassName, Test2Dataset
import os
from utils import evaluation_metrics as Metrics
import json
import pickle
import time

@torch.no_grad()
def eval_fine_ranking_on_testset1(model, cfg, logger=None):
    print_func = print if logger is None else logger
    model.eval()
    device = next(model.parameters()).device
    print_func('Fine ranking on TestSet1 ...')
    results = dict()
    eval_categories = ClassName[:cfg.num_classes]

    for c_idx, category in enumerate(eval_categories):
        scores = []
        bgs = []
        bg_fileNames = []
        fg_list = []
        ratio_list = []
        for bg_index in tqdm(range(20), desc="{} fine ranking".format(category)):
            fg_dataset = TestDataset(cfg, category=category, bg_index=bg_index)
            fg_loader  = DataLoader(fg_dataset, batch_size=cfg.eval_bs, shuffle=False,
                                   num_workers=cfg.num_workers, drop_last=False)
            bg_name = fg_dataset.bg_name
            bg_fileNames.append(bg_name)
            bgs.append([bg_name])
            fg_files = fg_dataset.fgs
            fg_list.append(fg_dataset.fgs)
            eachbg_score = []
            ratio_dif = []

            for sample in fg_loader:
                bg_im = sample['bg'].to(device)
                fg_im = sample['fg'].to(device)
                q_box  = sample['query_box'].to(device)
                c_box  = sample['crop_box'].to(device)
                output = model(bg_im, fg_im, q_box, c_box)[-1]
                eachbg_score.append(output)
                ratio_dif.extend(sample['ratio_dif'])
            eachbg_score = torch.cat(eachbg_score, dim=0)
            scores.append(eachbg_score)
            ratio_list.append(ratio_dif)

        scores = torch.stack(scores, dim=0)  # (num_bg, num_fg)
        scores = scores.cpu().numpy()

        labels = generate_label_matrix(fg_files, bgs)  # (num_bg, num_fg) with binary elements
        assert ((labels == 1).sum(1) < 1).sum() == 0, ((labels == 1).sum(1) < 1).sum()
        num_pos = (labels == 1).sum(1)
        print_func('{}: {} bg query, {} fg images, positive fg per bg: {}~{}'.format(
            category, labels.shape[0], labels.shape[1], num_pos.min(), num_pos.max()))
        rank_labels = Metrics.rank_labels_by_predicted_scores(labels, scores)
        results[category] = calc_evaluation_metrics(rank_labels.copy(), cfg.test1_metrics)

        ss = f'{c_idx}/{len(eval_categories)} {category}: '
        for k, v in results[category].items():
            ss += '{}={:.2f} | '.format(k, v)
        print_func(ss)
    results['overall'] = calc_overall_results(results)

    ss = 'Total {} categories, overall results: '.format(len(results) - 1)
    for k, v in results['overall'].items():
        ss += ' {}={:.2f} | '.format(k, v)
    print_func(ss)
    return results
    
@torch.no_grad()
def get_model_score_on_testset2(model, cfg, save_score=False, save_folder="", logger=None):
    print_func = print if logger is None else logger
    model.eval()
    device = next(model.parameters()).device
    print_func('Getting model score on TestSet2 ...')
    results = dict()
    eval_categories = ClassName[:cfg.num_classes]
    save_info = {}
    for c_idx, category in enumerate(eval_categories):
        save_info[category] = {}
        scores = []
        fg_list, label_list = [], []
        ratio_dif = []
        bg_names = []
        for bg_index in tqdm(range(20), desc="{} scores".format(category)):
            fg_dataset = Test2Dataset(cfg, category=category, bg_index=bg_index)
            fg_loader  = DataLoader(fg_dataset, batch_size=cfg.eval_bs, shuffle=False,
                                   num_workers=cfg.num_workers, drop_last=False)
            eachbg_score = []
            fg_label = []
            bg_names.append(fg_dataset.bg_name)
            ratio_list = []

            for sample in fg_loader:
                bg_im = sample['bg'].to(device)
                fg_im = sample['fg'].to(device)
                q_box  = sample['query_box'].to(device)
                c_box  = sample['crop_box'].to(device)
                output = model(bg_im, fg_im, q_box, c_box)[-1]
                eachbg_score.append(output)
                fg_label.extend(sample['label'])
                ratio_list.extend(sample['ratio_dif'])
            eachbg_score = torch.cat(eachbg_score, dim=0)
            scores.append(eachbg_score)
            fg_list.append(fg_dataset.fgs)
            label_list.append(fg_label)
            ratio_dif.append(ratio_list)

        scores = torch.stack(scores, dim=0)  # (num_bg, num_fg)
        scores = scores.cpu().numpy()
        
        save_info[category]["bgs"] = bg_names
        save_info[category]["fgs"] = fg_list
        save_info[category]["labels"] = label_list
        save_info[category]["scores"] = scores
        save_info[category]["ratio_dif"] = ratio_dif

    if save_score == True and save_folder != "":
        save_dir = os.path.join(PROJ_dir, save_folder)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "{}_Ours_testset2_scores.pkl".format(time.strftime('%Y%m%d_%H%M%S')))
        with open(save_path, "wb") as f:
            pickle.dump(save_info, f)
    
    return save_info

@torch.no_grad()
def eval_score_shape_ranking_on_testset2(cfg, thre, save_info, save_top20=False, save_folder="", logger=None):
    print_func = print if logger is None else logger
    print_func('Fine ranking on TestSet2 ...')
    results = dict()
    eval_categories = ClassName[:cfg.num_classes]
    json_data = {}
    for category in eval_categories:
        
        bg_fileNames = save_info[category]["bgs"]
        scores = save_info[category]["scores"]
        fg_list = save_info[category]["fgs"]
        label_list = save_info[category]["labels"]
        ratio_dif = save_info[category]["ratio_dif"]
        
        ratio_np = np.array(ratio_dif)
        ratio_flag = ratio_np < thre
        scores = scores * ratio_flag        
        
        json_data[category] = get_top_info(bg_fileNames, fg_list, scores.copy())

        labels = generate_label_matrix_v2(fg_list, label_list)  # (num_bg, num_fg) with binary elements
        assert ((labels == 1).sum(1) < 1).sum() == 0, ((labels == 1).sum(1) < 1).sum()
        num_pos = (labels == 1).sum(1)
        print_func('Total {} bg query, {} fg images, positive fg per bg: {}~{}'.format(
            labels.shape[0], labels.shape[1], num_pos.min(), num_pos.max()))
        rank_labels = Metrics.rank_labels_by_predicted_scores(labels, scores)
        results[category] = calc_evaluation_metrics_onlyValid(rank_labels.copy(), cfg.test2_metrics, ratio_flag.copy())

        ss = '{}: '.format(category)
        for k, v in results[category].items():
            ss += '{}={:.2f} | '.format(k, v)
        print_func(ss)
    if save_top20 and save_folder != "":
        save_dir = os.path.join(PROJ_dir, save_folder)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "{}_Ours_threshold{}_top20.json".format(time.strftime('%Y%m%d_%H%M%S'), str(thre)))
        with open(save_path, "w") as f:
            json.dump(json_data, f)
    results['overall'] = calc_overall_results(results)

    ss = 'Total {} categories, overall results: '.format(len(results) - 1)
    for k, v in results['overall'].items():
        ss += ' {}={:.2f} | '.format(k, v)
    print_func(ss)
    return results
  
def get_top_info(bg_names, fg_list, scores):
    indices   = (-scores).argsort(axis=1)
    infos = {}
    for i in range(len(bg_names)):
        bg = bg_names[i]
        infos[bg] = []
        for k in range(20):
            infos[bg].append(fg_list[i][indices[i][k]])
    return infos

def calc_overall_results(results):
    overall = {}
    for category in results.keys():
        for metric in results[category].keys():
            if metric in overall:
                overall[metric].append(results[category][metric])
            else:
                overall[metric] = [results[category][metric]]
    for metric in overall.keys():
        overall[metric] = np.mean(overall[metric])
    return overall

def calc_evaluation_metrics(rank_labels, metrics):
    results = dict()
    for m in metrics:
        if m == 'mAP':
            results[m] = Metrics.mean_average_precision(rank_labels, -1)
        elif 'mAP-' in m:
            k = int(m.split('mAP-')[-1])
            results[m] = Metrics.mean_average_precision(rank_labels, k)
        elif 'Precision@' in m:
            suffix = m.split('Precision@')[-1]
            if '%' in suffix:
                percent = int(suffix.split('%')[0]) * 0.01
                k = int(percent * rank_labels.shape[-1])
            else:
                k = int(suffix)
            results[m] = Metrics.precision_at_k(rank_labels, k)
        elif 'Recall@' in m:
            suffix = m.split('Recall@')[-1]
            if '%' in suffix:
                percent = int(suffix.split('%')[0]) * 0.01
                k = int(percent * rank_labels.shape[-1])
            else:
                k = int(suffix)
            results[m] = Metrics.recall_at_k(rank_labels, k)
        else:
            raise Exception('Undefined evaluation metric: {}'.format(m))
    return results

def calc_evaluation_metrics_onlyValid(rank_labels, metrics, flags):
    flags = np.sum(flags, axis=1)
    results = dict()
    for m in metrics:
        if m == 'mAP':
            results[m] = Metrics.mean_average_precision_onlyValid(rank_labels, flags.copy(), -1)
        elif 'mAP-' in m:
            k = int(m.split('mAP-')[-1])
            results[m] = Metrics.mean_average_precision_onlyValid(rank_labels, flags.copy(), k)
        elif 'Precision@' in m:
            suffix = m.split('Precision@')[-1]
            if '%' in suffix:
                percent = int(suffix.split('%')[0]) * 0.01
                k = int(percent * rank_labels.shape[-1])
            else:
                k = int(suffix)
                
            results[m] = Metrics.precision_at_k_onlyValid(rank_labels, flags.copy(), k)
        elif 'Recall@' in m:
            suffix = m.split('Recall@')[-1]
            if '%' in suffix:
                percent = int(suffix.split('%')[0]) * 0.01
                k = int(percent * rank_labels.shape[-1])
            else:
                k = int(suffix)
            results[m] = Metrics.recall_at_k(rank_labels, k)
        else:
            raise Exception('Undefined evaluation metric: {}'.format(m))
    return results

def generate_label_matrix(fg_files, pos_fgs):
    '''
    :param fg_files: list with length of num_fg
    :param pos_fgs:  list with length of num_bg
    :return: binary matrix: num_bg x num_fg
    '''
    labels   = np.zeros((len(pos_fgs), len(fg_files)))
    for i in range(len(pos_fgs)):
        labels[i] = np.isin(fg_files, pos_fgs[i])
        assert labels[i].sum() >= 1, \
             '{} {}'.format(fg_files, pos_fgs[i])
    return labels

def generate_label_matrix_v2(fg_files, label_list):
    '''
    :return: binary matrix: num_bg x num_fg
    '''
    labels   = np.array(label_list).reshape((len(label_list), -1))
    # check if order of fg list is same
    assert len(np.unique(fg_files, axis=0)) == 1, np.unique(fg_files, axis=0)
    return labels