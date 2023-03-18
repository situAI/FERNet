import torch
from core.data import build_dataloader
from core.model import build_model
from core.optimizer import build_optimizer, build_lr_scheduler
from core.loss import build_loss
from core.metric import build_metric
from utils.registery import SOLVER_REGISTRY
from utils.logger import get_logger_and_log_path
from .base_solver import BaseSolver
import os
import copy
import datetime
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import pandas as pd
import yaml
import joblib
import optuna
import pandas as pd


@SOLVER_REGISTRY.register()
class OptunaSolver(object):
    def __init__(self, init_config):
        self.init_config = init_config
        self.txt_file = None
        self.log_root = os.path.join(self.init_config['solver']['logger']['log_root'], self.init_config['solver']['logger']['suffix'])
        self.open_txt_file()
        if self.init_config['task'] == 'va':
            self.txt_file.write(f"lr,optimizer,batch_size,nlayers,dropout,seq_len,valence,arousal\n")
        else:
            self.txt_file.write(f"lr,optimizer,batch_size,nlayers,dropout,seq_len,F1\n")
        self.close_txt_file()
    
    def open_txt_file(self):
        self.txt_file = open(os.path.join(self.log_root, f"optuna_{self.init_config['task']}.csv"), 'w', buffering=1)

    def close_txt_file(self):
        self.txt_file.close()


    def objective(self, trial):

        cfg = copy.deepcopy(self.init_config)

        lr = trial.suggest_loguniform('lr', 1e-7, 1e-2)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adagrad", "AdamW"])
        batch_size = trial.suggest_int('batch_size', 8, 48, step=8)
        nlayers = trial.suggest_int('nlayers', 1, 12, step=1)
        dropout = trial.suggest_float('dropout', 0, 0.6, step=0.1)
        seq_len = trial.suggest_categorical("seq_len", [32, 64, 128, 256])

        cfg['solver']['optimizer']['args']['lr'] = lr
        cfg['solver']['optimizer']['args']['weight_decay'] = lr * 0.1
        cfg['solver']['optimizer']['name'] = optimizer_name
        cfg['dataloader']['batch_size'] = batch_size
        cfg['model']['args']['nlayers'] = nlayers
        cfg['model']['args']['dropout'] = dropout
        cfg['model']['args']['seq_len'] = seq_len
        cfg['train_data']['args']['seq_len'] = seq_len
        cfg['val_data']['args']['seq_len'] = seq_len

        solver = BaseSolver(cfg)

        if cfg['task'] == 'va':
            best_v, best_a = solver.train()

            self.open_txt_file()
            self.txt_file.write(f"{lr},{optimizer_name},{batch_size},{nlayers},{dropout},{seq_len},{best_v},{best_a}\n")
            self.close_txt_file()

            return best_v, best_a

        else:
            out = solver.train()

            self.open_txt_file()
            self.txt_file.write(f"{lr},{optimizer_name},{batch_size},{nlayers},{dropout},{seq_len},{out}\n")
            self.close_txt_file()

            return out


    def run(self):
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        os.makedirs(self.log_root, exist_ok=True)
        joblib.dump(study, os.path.join(self.log_root, f"{self.init_config['task']}_study.pkl"))
        study.optimize(self.objective, n_trials=120)
        joblib.dump(study, os.path.join(self.log_root, f"{self.init_config['task']}_study.pkl"))
        print(f'Best params: {study.best_params}')
        print(f'Best value: {study.best_value}')

