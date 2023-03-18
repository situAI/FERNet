import torch
from core.data import build_dataloader
from core.model import build_model
from core.metric import build_metric
from scipy.stats import pearsonr

import os
import copy
import random
import datetime
import numpy as np
import pandas as pd
import yaml

def init_seed(seed=778):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

model_dir = r'/data1/ERI/submit_model/dim_max/'
cfg = yaml.load(open(os.path.join(model_dir, 'config_0125.yaml'), 'r').read(), Loader=yaml.FullLoader)
output_path = r'results/eri_all_fau_7_expr_csm1s_112_0.15000000000000002.txt'

output_file = open(output_path, 'w')

init_seed(cfg['seed'])
torch.distributed.init_process_group(backend='nccl')

task = cfg['task']
_, val_loader = build_dataloader(cfg)
len_val_loader = len(val_loader)
model = build_model(cfg)
model.load_state_dict(torch.load(os.path.join(model_dir, 'ckpt_eri_1.pth')))
model = model.cuda()
model.eval()

hyper_params = cfg['solver']['args']
crt_date = datetime.date.today().strftime('%Y-%m-%d')
metric_fn = build_metric(cfg)

torch.no_grad()
preds = None 
labels = None
for i, data in enumerate(val_loader):
    feat = data['feat'].cuda()
    label = data['label'].cpu().numpy()
    audio_feat = data['audio_feat'].cuda()
    video_name = data['name']

    # compute output
    output = model(feat, audio_feat)

    for o, output_ in  enumerate(output):
        output_file.write(video_name[o])
        output_ = output_.cpu().detach().numpy()
        for j in output_:
            output_file.write(' ' + str(j))
        output_file.write('\n')

    output = output.cpu().detach().numpy()
    output = np.maximum(output, 0)
    # new_output = []
    # for x in output:
    #     x = np.where(x==np.amax(x), 1, x)
    #     new_output.append(x)
    # output = np.array(new_output)
    print(output)
    preds = output if preds is None else np.concatenate([preds, output], axis=0)
    labels = label if labels is None else np.concatenate([labels, label], axis=0)

# my_rho = np.corrcoef(preds, labels)
pccs = 0
for i in range(7):
    pccs_ = pearsonr(preds[:,i], labels[:, i])
    pccs += pccs_.statistic

print(pccs/7)

output_file.close()


