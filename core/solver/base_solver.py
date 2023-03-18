import torch
from core.data import build_dataloader
from core.model import build_model
from core.optimizer import build_optimizer, build_lr_scheduler
from core.loss import build_loss
from core.metric import build_metric
from utils.registery import SOLVER_REGISTRY
from utils.helper import format_print_dict
from utils.logger import get_logger_and_log_path
import os
import copy
import datetime
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import pandas as pd
import yaml


@SOLVER_REGISTRY.register()
class BaseSolver(object):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.task = cfg['task']
        self.local_rank = torch.distributed.get_rank()
        self.train_loader, self.val_loader = build_dataloader(cfg)
        self.len_train_loader, self.len_val_loader = len(self.train_loader), len(self.val_loader)
        self.criterion = build_loss(cfg).cuda(self.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg))
        self.model = DistributedDataParallel(model.cuda(self.local_rank), device_ids=[self.local_rank], find_unused_parameters=True)
        self.optimizer = build_optimizer(cfg)(self.model.parameters(), **cfg['solver']['optimizer']['args'])
        self.hyper_params = cfg['solver']['args']
        crt_date = datetime.date.today().strftime('%Y-%m-%d')
        self.logger, self.log_path = get_logger_and_log_path(crt_date=crt_date, **cfg['solver']['logger'])
        self.metric_fn = build_metric(cfg)
        try:
            self.epoch = self.hyper_params['epoch']
        except Exception:
            raise 'should contain epoch in {solver.args}'
        if self.local_rank == 0:
            self.save_dict_to_yaml(self.cfg, os.path.join(self.log_path, 'config.yaml'))
            self.logger.info(self.cfg)

    def train(self):
        if torch.distributed.get_rank() == 0:
            self.logger.info('==> Start Training')
        lr_scheduler = build_lr_scheduler(self.cfg)(self.optimizer, **self.cfg['solver']['lr_scheduler']['args'])

        val_v_list = [-1]
        val_a_list = [-1]
        val_peek_list = [-1]

        for t in range(self.epoch):
            self.train_loader.sampler.set_epoch(t)
            if torch.distributed.get_rank() == 0:
                self.logger.info(f'==> epoch {t + 1}')
            self.model.train()

            pred_list = list()
            label_list = list()
            mean_loss = 0.0

            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                feat = data['feat'].cuda(self.local_rank)
                label = data['label'].cuda(self.local_rank)
                label_ass = label[:,1]+label[:,5]
                label_ass = label_ass.view(label.shape[0], -1)

                pred, ass_pred = self.model(feat)

                bs, _ = pred.shape

                loss = self.criterion(pred, label)
                mean_loss += loss.item()

                if (i == 0 or i % 100 == 0) and (torch.distributed.get_rank() == 0):
                    self.logger.info(f'epoch: {t + 1}/{self.epoch}, iteration: {i + 1}/{self.len_train_loader}, loss: {loss.item() :.4f}')
                
                loss.backward()
                self.optimizer.step()

                batch_pred = [torch.zeros_like(pred) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_pred, pred)
                pred_list.append(torch.cat(batch_pred, dim=0).detach().cpu())

                batch_label = [torch.zeros_like(label) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_label, label)
                label_list.append(torch.cat(batch_label, dim=0).detach().cpu())

            pred_list = torch.cat(pred_list, dim=0).cuda()
            label_list = torch.cat(label_list, dim=0).cuda()
            # pred_list = pred_list.numpy()
            # label_list = label_list.numpy()
            metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
            mean_loss = mean_loss / self.len_train_loader

            print_dict = dict()
            print_dict.update({'epoch': f'{t + 1}/{self.epoch}'})
            print_dict.update({'mean_loss': mean_loss})
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            print_dict.update(metric_dict)

            print_str = format_print_dict(print_dict)

            if torch.distributed.get_rank() == 0:
                self.logger.info(f"==> train: {print_str}")
                if self.task == 'va':
                    peek_v, peek_a = self.validate(t + 1)
                    if peek_v >= max(val_v_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'valence')
                        val_v_list.append(peek_v)
                    if peek_a >= max(val_a_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'arousal')
                        val_a_list.append(peek_a)
                else:
                    peek = self.validate(t + 1)
                    # if peek > max(val_peek_list):
                    self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task)
                    val_peek_list.append(peek)

            lr_scheduler.step()

        if self.local_rank == 0:
            if self.task == 'va':
                self.logger.info(f'==> End Training, BEST Valence: {max(val_v_list)}, BEST Arousal: {max(val_a_list)}')

                return max(val_v_list), max(val_a_list)
            else:
                self.logger.info(f'==> End Training, BEST F1: {max(val_peek_list)}')

                return max(val_peek_list)

    @torch.no_grad()
    def val(self, t):
        self.model.eval()

        pred_list = list()
        label_list = list()

        for i, data in enumerate(self.val_loader):
            feat = data['feat'].cuda(self.local_rank)
            label = data['label'].cuda(self.local_rank)

            pred, _ = self.model(feat)

            pred_list.append(pred.detach().cpu())
            label_list.append(label.detach().cpu())

        pred_list = torch.cat(pred_list, dim=1)
        label_list = torch.cat(label_list, dim=1)
        seq_len, bs = pred_list.shape[:2]
        pred_list = pred_list.reshape((seq_len * bs, -1))
        label_list = label_list.reshape((seq_len * bs, -1))
        pred_list = pred_list.numpy()
        label_list = label_list.numpy()

        metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
        print_dict = dict()
        print_dict.update({'epoch': t})
        print_dict.update(metric_dict)
        print_str = format_print_dict(print_dict)

        if torch.distributed.get_rank() == 0:
            self.logger.info(f"==> val: {print_str}")

        if self.task == 'va':
            peek_v = metric_dict['valence_ccc']
            peek_a = metric_dict['arousal_ccc']
            return peek_v, peek_a
        else:
            peek = metric_dict['F1']
            return peek

    @torch.no_grad()
    def validate(self, t):
        # switch to evaluate mode
        self.model.eval()

        preds = None 
        labels = None
        for i, data in enumerate(self.val_loader):
            feat = data['feat'].cuda(self.local_rank)
            label = data['label'].cuda(self.local_rank)

            # compute output
            output, _ = self.model(feat)
            preds = output if preds is None else torch.cat([preds, output], dim=0)
            labels = label if labels is None else torch.cat([labels, label], dim=0)

        # print(output)
        metric_dict = self.metric_fn(**{'pred': preds, 'gt': labels})
        print_dict = dict()
        print_dict.update({'epoch': t})
        print_dict.update(metric_dict)
        print_str = format_print_dict(print_dict)

        if torch.distributed.get_rank() == 0:
            self.logger.info(f"==> val: {print_str}")

        if self.task == 'va':
            peek_v = metric_dict['valence_ccc']
            peek_a = metric_dict['arousal_ccc']
            return peek_v, peek_a
        elif self.task == 'eri':
            peek = metric_dict['avg_p']
            return peek
        else:
            peek = metric_dict['F1']
            return peek

    def run(self):
        self.train()

    @staticmethod
    def save_dict_to_yaml(dict_value, save_path):
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(dict_value, file, sort_keys=False)


    def save_checkpoint(self, model, cfg, log_path, epoch_id, task_name):
        model.eval()
        torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_{task_name}_{epoch_id}.pth'))
