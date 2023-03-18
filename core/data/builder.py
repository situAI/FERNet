import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.registery import DATASET_REGISTRY, COLLATE_FN_REGISTRY
from .collate_fn import base_collate_fn

from .SequenceData import SequenceData
from .ERIData import ERIData


def build_dataset(cfg, prefix):

    dataset_cfg = copy.deepcopy(cfg)
    try:
        dataset_cfg = dataset_cfg[prefix]
    except Exception:
        raise f'should contain {prefix}!'


    data = DATASET_REGISTRY.get(dataset_cfg['name'])(**dataset_cfg['args'])

    return data



def build_dataloader(cfg):

    dataloader_cfg = copy.deepcopy(cfg)
    try:
        dataloader_cfg = cfg['dataloader']
    except Exception:
        raise 'should contain {dataloader}!'

    train_ds = build_dataset(cfg, 'train_data')
    val_ds = build_dataset(cfg, 'val_data')

    train_sampler = DistributedSampler(train_ds)
    collate_fn = COLLATE_FN_REGISTRY.get(dataloader_cfg.pop('collate_fn'))

    train_loader = DataLoader(train_ds,
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              **dataloader_cfg)

    val_loader = DataLoader(val_ds,
                            collate_fn=collate_fn,
                            **dataloader_cfg)

    return train_loader, val_loader
