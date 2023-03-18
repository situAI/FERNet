import torch
from core.data import build_dataloader
from core.model import build_model
from core.optimizer import build_optimizer, build_lr_scheduler
from core.loss import build_loss
from core.metric import build_metric
from utils.registery import SOLVER_REGISTRY
from utils.logger import get_logger_and_log_path
import os
import copy
import datetime
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import pandas as pd
import yaml
from .base_solver import BaseSolver


@SOLVER_REGISTRY.register()
class RDropSolver(BaseSolver):
    def __init__(self, cfg):
        super().__init__(cfg)

    def train(self):
        if self.task == 'va':
            raise 'RDrop not implement in task: `va` yet!'
        if torch.distributed.get_rank() == 0:
            self.logger.info('==> Start Training')
        lr_scheduler = build_lr_scheduler(self.cfg)(self.optimizer, **self.cfg['solver']['lr_scheduler']['args'])

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

                logits1 = self.model(feat)
                logits2 = self.model(feat)


                seq_len, bs, _ = logits1.shape
                logits1 = logits1.reshape((seq_len * bs, -1))
                logits2 = logits2.reshape((seq_len * bs, -1))
                label = label.reshape((seq_len * bs, -1))

                loss = self.criterion(logits1, logits2, label)
                mean_loss += loss.item()

                if self.task == 'expr':
                    pred = logits1.argmax(dim=-1)
                else:
                    pred = logits1

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

            pred_list = torch.cat(pred_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            pred_list = pred_list.numpy()
            label_list = label_list.numpy()
            if self.task == 'expr':
                label_list = label_list.argmax(axis=-1)
            metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list}, class_num=self.cfg['model']['args']['out_dim'])
            mean_loss = mean_loss / self.len_train_loader

            print_dict = dict()
            print_dict.update({'epoch': f'{t + 1}/{self.epoch}'})
            print_dict.update({'mean_loss': mean_loss})
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            print_dict.update(metric_dict)

            if torch.distributed.get_rank() == 0:
                self.logger.info(f"==> train: {print_dict}")
                if self.task == 'va':
                    peek_v, peek_a = self.val(t + 1)
                    if peek_v >= max(val_v_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'valence')
                        val_v_list.append(peek_v)
                    if peek_a >= max(val_a_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'arousal')
                        val_a_list.append(peek_a)
                else:
                    peek = self.val(t + 1)
                    if peek > max(val_peek_list):
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

