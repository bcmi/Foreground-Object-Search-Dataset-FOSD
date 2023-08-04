import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import shutil
from torch.utils.data import DataLoader, DistributedSampler
import csv
from einops import rearrange, repeat
import torch.optim as optim
from network.networks import StudentModel, SingleScaleD
from dataset.datasets import TrainDataset, ClassName
from evaluate.evaluate import *
from utils import loss_function, training_tools
from utils.logger import get_logger
from options.options import parse_args
from config.config import Config
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy

logger = None

def build_dataloader(cfg):
    dataset = TrainDataset(cfg)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            drop_last=False, sampler=sampler)
    if cfg.local_rank == 0:
        logger.info('FOS Dataset contains {} samples, batch size={}x{}, total of {} batches'.format(
            len(dataset), cfg.batch_size, torch.cuda.device_count(), len(dataloader)
        ))
    return dataloader


class Trainer(object):
    def __init__(self, model_list, cfg):
        self.cfg = cfg
        self.student, self.teacher = model_list
        self.device = next(self.student.parameters()).device
        self.dataloader = build_dataloader(cfg)
        self.create_optimizer()
        self.epoch = 0
        self.iters = 0
        self.training_loss = ['TripletLoss', 'ClsLoss', 'KDLoss']
        self.avg_loss = {k: [] for k in self.training_loss}
        self.smooth_loss = None
        self.smooth_coe  = 0.4
        self.test1_metrics = cfg.test1_metrics
        self.test2_metrics = cfg.test2_metrics

        self.fine_test1   = self.create_results_dict(self.test1_metrics)
        self.fine_test2   = self.create_results_dict(self.test2_metrics)

        self.test_epochs  = []
        self.best_results = dict()
        for k in self.test1_metrics:
            self.best_results[k] = 0
        for k in self.test2_metrics:
            self.best_results[k] = 0
        if self.cfg.local_rank == 0:
            self.writer = self.create_writer()
        self.triplet_loss = loss_function.TripletLoss(cfg.triplet_margin)

    def create_results_dict(self, metrics):
        results = dict()
        for cls in ClassName[:cfg.num_classes] + ['overall']:
            results[cls] = {k: [] for k in metrics}
        return results

    def create_writer(self):
        logger.info('Create tensorboardX writer...')
        writer = SummaryWriter(log_dir=self.cfg.log_dir)
        return writer

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.cfg.E_lr,
                                    betas=(0.5, 0.999), weight_decay=1e-4)

    def save_checkpoints(self, suffix):
        if self.cfg.local_rank == 0:
            checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f'{suffix}.pth')
            torch.save(self.student.module.state_dict(), checkpoint_path)

    def run(self):
        for epoch in range(self.cfg.max_epoch):
            self.dataloader.sampler.set_epoch(epoch)
            self.run_epoch()
            self.epoch += 1

            if self.cfg.local_rank == 0:
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.epoch)
                if self.cfg.save_freq > 0 and (self.epoch % self.cfg.save_freq == 0):
                    self.save_checkpoints(f'epoch{self.epoch}')
                    logger.info('Save checkpoint...')

                if (self.epoch % self.cfg.test_freq == 0 or epoch in [0, self.cfg.max_epoch-1]) and (self.cfg.eval_testset1 or self.cfg.eval_testset2):
                    epoch_result1, epoch_result2 = self.eval_training()
                    if self.cfg.eval_testset1:
                        for m in self.test1_metrics:
                            update = False
                            if epoch_result1['overall'][m] > self.best_results[m]:
                                update = True
                            if update:
                                self.best_results[m] = epoch_result1['overall'][m]
                                self.save_checkpoints(f'best-{m}')
                                logger.info('Update best {} model, best {}={:.2f}'.format(m, m, self.best_results[m]))
                            if m in ['Recall@1', 'mAP-20']:
                                self.writer.add_scalar(f'Test/Best_{m}', self.best_results[m], self.epoch)
                    if self.cfg.eval_testset2:
                        for m in self.test2_metrics:
                            update = False
                            if epoch_result2['overall'][m] > self.best_results[m]:
                                update = True
                            if update:
                                self.best_results[m] = epoch_result2['overall'][m]
                                self.save_checkpoints(f'best-{m}')
                                logger.info('Update best {} model, best {}={:.2f}'.format(m, m, self.best_results[m]))
                            if m in ['Recall@1', 'mAP-20']:
                                self.writer.add_scalar(f'Test/Best_{m}', self.best_results[m], self.epoch)
                dist.barrier()
            else:
                dist.barrier()

    def update_results_dict(self, src, dst):
        tmp = copy.deepcopy(src)
        for k,v in dst.items():
            if isinstance(v, dict):
                for k1,v1 in v.items():
                    tmp[k][k1].append(v1)
            else:
                tmp[k].append(v)
        return tmp

    def eval_training(self):
        self.test_epochs.append(self.epoch)

        fine_test1 = None
        fine_test2 = None

        if self.cfg.eval_testset1:
            fine_test1   = eval_fine_ranking_on_testset1(self.student, cfg, logger.info)
            self.fine_test1 = self.update_results_dict(self.fine_test1, fine_test1)
            self.write2csv(self.fine_test1, self.test1_metrics, '-testset1')

        if self.cfg.eval_testset2:
            save_info = get_model_score_on_testset2(self.student, cfg, logger.info)
            fine_test2 = eval_score_shape_ranking_on_testset2(self.cfg, 1.2, save_info)
            self.fine_test2 = self.update_results_dict(self.fine_test2, fine_test2)
            self.write2csv(self.fine_test2, self.test2_metrics, '-testset2')

        return fine_test1, fine_test2

    def write2csv(self, test_results, test_metrics, suffix=''):
        csv_path = os.path.join(self.cfg.exp_path, '{}{}.csv'.format(self.cfg.exp_name, suffix))
        metrics  = copy.deepcopy(test_metrics)
        results  = test_results['overall']
        for cls in test_results.keys():
            if cls == 'overall':
                continue
            for k in test_metrics:
                tmp_k = f'{cls}-{k}'
                try:
                    results[tmp_k] = test_results[cls][k]
                except:
                    logger.info([cls, k, list(test_results.keys()), list(test_results[cls].keys())])
                metrics.append(tmp_k)
        header  = ['epoch'] + metrics
        epoches = self.test_epochs
        results['epoch'] = epoches
        rows = [header]
        for i in range(len(epoches)):
            row = [results[m][i] for m in header]
            rows.append(row)

        for name in header:
            if name not in test_metrics:
                continue
            cur_result = results[name]
            best_index = cur_result.index(max(cur_result))
            title = 'best {} (epoch-{})'.format(name, epoches[best_index])
            row = [results[k][best_index] for k in header]
            row[0] = title
            rows.append(row)

        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)

        logger.info('Save result to ' + csv_path)


    def set_input(self, sample):
        # fetch data
        self.bg_im = sample['bg'].to(self.device)
        self.bs = self.bg_im.shape[0]

        pos_fg = sample['pos_fg'].to(self.device)
        pos_label = torch.ones(pos_fg.shape[:2], dtype=torch.float32).to(self.device)
        self.num_pos = pos_fg.shape[1]
        neg_fg = sample['neg_fg'].to(self.device)
        neg_label = torch.zeros(neg_fg.shape[:2], dtype=torch.float32).to(self.device)
        self.num_neg = neg_fg.shape[1]
        all_fg = torch.cat([pos_fg, neg_fg], dim=1)
        self.num_fg = all_fg.shape[1]
        self.all_fg = rearrange(all_fg, 'b n c h w -> (b n) c h w')
        self.fg_label = torch.cat([pos_label, neg_label], dim=1).reshape(-1)

        pos_comp = sample['pos_comp'].to(self.device)
        pos_scale_comp = sample['pos_scale_comp'].to(self.device)
        neg_comp = sample['neg_comp'].to(self.device)
        neg_scale_comp = sample['neg_scale_comp'].to(self.device)
        all_comp = torch.cat([pos_comp, neg_comp], dim=1)
        self.all_comp  = rearrange(all_comp, 'b n c h w -> (b n) c h w')
        all_scale_comp = torch.cat([pos_scale_comp, neg_scale_comp], dim=1)
        self.all_scale_comp = rearrange(all_scale_comp, 'b n c h w -> (b n) c h w')

        query_box   = sample['query_box'].to(self.device)
        self.query_box = repeat(query_box, '(b 1) d -> (b n) d', n=self.num_fg)

        crop_box = sample['crop_box'].to(self.device)
        self.crop_box = repeat(crop_box, '(b 1) d -> (b n) d', n=self.num_fg)

    def forward(self):
        repeat_bg = repeat(self.bg_im, '(b 1) c h w -> (b n) c h w', n=self.num_fg)
        self.bg_emb, self.fg_emb, self.pre_score = \
            self.student(repeat_bg, self.all_fg, self.query_box, self.crop_box, self.bs)

    def backward(self):
        triplet_loss = self.student.module.calcu_triplet_loss(self.fg_label, self.bs)
        self.avg_loss['TripletLoss'].append(triplet_loss.item())
        total_loss = triplet_loss
        if not self.cfg.distill_type == 'none':
            with torch.no_grad():
                scl_feat, score = self.teacher(self.all_scale_comp)
                kd_loss = self.student.module.calcu_distillation_loss(scl_feat)
        else:
            kd_loss = torch.zeros_like(total_loss)
        self.avg_loss['KDLoss'].append(kd_loss.item())
        total_loss += self.cfg.distill_weight * kd_loss

        cls_loss = self.student.module.calcu_classify_loss(self.fg_label)
        self.avg_loss['ClsLoss'].append(cls_loss.item())
        total_loss += self.cfg.classify_weight * cls_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def print_training_process(self, batch_idx):
        start = self.epoch_start
        for k, v in self.avg_loss.items():
            self.avg_loss[k] = sum(v) / len(v)

        if self.smooth_loss != None:
            for k in self.avg_loss.keys():
                self.avg_loss[k] = (1 - self.smooth_coe) * self.avg_loss[k] + \
                                   self.smooth_coe * self.smooth_loss[k]

        time_per_batch = (time.time() - start) / (batch_idx + 1.)
        last_batches = (self.cfg.max_epoch - self.epoch - 1) * self.total_batch + \
                       (self.total_batch - batch_idx - 1)
        last_time = int(last_batches * time_per_batch)
        time_str = str(datetime.timedelta(seconds=last_time))
        out_ss = '=== epoch:{}/{} | batch:{}/{} | lr:{:.6f} | estimated remaining time:{} ==='.format(
            self.epoch, self.cfg.max_epoch, batch_idx + 1, self.total_batch, self.cur_lr, time_str)
        if self.cfg.local_rank == 0:
            logger.info(out_ss)

        out_ss = ' '
        for k, v in self.avg_loss.items():
            out_ss += ' {}:{:.4f} |'.format(k, v)
            if self.cfg.local_rank == 0:
                self.writer.add_scalar(f'Train/{k}', v, self.iters)
        if self.cfg.local_rank == 0:
            logger.info(out_ss)
        self.smooth_loss = self.avg_loss
        for k in self.avg_loss.keys():
            self.avg_loss[k] = []

    def run_epoch(self):
        if self.teacher != None:
            self.teacher.eval()
        self.student.train()
        self.cur_lr = self.optimizer.param_groups[0]['lr']
        self.total_batch = len(self.dataloader)
        self.epoch_start = time.time()

        for batch_idx, sample in enumerate(self.dataloader):
            self.set_input(sample)
            self.forward()
            self.backward()

            if self.iters % self.cfg.display_freq == 0:
                self.print_training_process(batch_idx)
            self.iters += 1


if __name__ == '__main__':
    config_file, local_rank, config_info = parse_args()
    cfg = Config(config_file)
    cfg.local_rank = local_rank

    cfg.generate_path()
    if cfg.local_rank == 0:
        cfg.print_yaml_params()
        cfg.create_path()

    training_tools.set_seed(cfg.local_rank)
    world_size = torch.cuda.device_count()
    print('rank{} distribution initialization ...'.format(cfg.local_rank))
    device = training_tools.dist_init(cfg.local_rank, world_size, cfg.log_dir)
    torch.cuda.set_device(device)
    print('rank{} distribution initialization completed ...'.format(cfg.local_rank))

    if cfg.local_rank == 0:
        logger = get_logger(os.path.join(cfg.exp_path, 'training.log'))
        # backup experiment code
        for path in ['network/networks.py', 'dataset/datasets.py',
                     'evaluate/evaluate.py', os.path.abspath(__file__), config_file]:
            if not os.path.exists(path):
                path = os.path.join('..', path)
            assert os.path.exists(path), path
            if os.path.isdir(path):
                tar_path = os.path.join(cfg.code_dir, os.path.basename(path))
                shutil.copytree(path, tar_path)
            else:
                if os.path.basename(path).endswith(('.yaml', '.py')):
                    tar_path = os.path.join(cfg.code_dir, os.path.basename(path))
                    shutil.copy(path, tar_path)
            logger.info(f'backup {path}')
        dist.barrier()
    else:
        dist.barrier()

    Student = StudentModel(cfg).to(device)
    Student = DDP(Student)

    if cfg.distill_type not in ['none', 'onlyclassify']:
        Teacher = SingleScaleD(False)
        ckpt_dir = os.path.join(PROJ_dir, cfg.teacher_path)
        weight_file = os.path.join(ckpt_dir, cfg.teacher_model)
        assert os.path.exists(weight_file), weight_file
        if cfg.local_rank == 0:
            logger.info(f'load teacher model from {weight_file}')
        Teacher.load_state_dict(torch.load(weight_file, map_location='cpu'))
        Teacher.to(device).eval()
    else:
        Teacher = None
    trainer = Trainer([Student, Teacher], cfg)
    trainer.run()
    if cfg.local_rank == 0:
        dist.destroy_process_group()