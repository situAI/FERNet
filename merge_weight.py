import torch
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd

def average_checkpoints(filenames: List[Path], device: torch.device = torch.device("cpu")) -> dict:
    """Average a list of checkpoints.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device)
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg

model_1_path = r'/data1/ERI/submit_model/train_val_mix/ckpt_eri_5.pth'
model_2_path = r'/data1/ERI/submit_model/train_val_mix/ckpt_eri_6.pth'
model_3_path = r'/data1/ERI/submit_model/train_val_mix/ckpt_eri_7.pth'
save_model_path = r'/data1/ERI/submit_model/train_val_mix/train_val_mix.pth'

model_path_list = [model_1_path, model_2_path, model_3_path]

model = average_checkpoints(model_path_list)
torch.save(model, save_model_path)
