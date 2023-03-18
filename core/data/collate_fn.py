import torch
from utils.registery import COLLATE_FN_REGISTRY
import numpy as np


@COLLATE_FN_REGISTRY.register()
def base_collate_fn(batch):
    feats, labels, audio_feats, names = list(), list(), list(), list()
    for crt_feat, crt_label, name in batch:
        feats.append(crt_feat)
        labels.append(crt_label)
        names.append(name)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    labels = torch.from_numpy(np.asarray(labels))

    return {'feat': feats, 'label': labels, 'name': names}


@COLLATE_FN_REGISTRY.register()
def mixup_collate_fn(batch):
    feats, labels, audio_feats = list(), list(), list()
    for crt_feat, crt_label in batch:
        if len(labels)>0 and np.random.random() < 0.5:
            if abs(np.sum(crt_label - labels[-1])) > len(crt_label)//4:
                mixup(feats[-1], crt_feat, labels[-1], crt_label)
        feats.append(crt_feat)
        labels.append(crt_label)
        audio_feats.append(crt_audio_feat)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    labels = torch.from_numpy(np.asarray(labels)).transpose(0, 1)
    audio_feats = torch.from_numpy(np.asarray(audio_feats)).transpose(0, 1)

    return {'feat': feats, 'label': labels, 'audio_feat': audio_feats}


def mixup(feat1, feat2, label1, label2, ratio=0.2):
    for idx, (f1, f2, l1, l2) in enumerate(zip(feat1, feat2, label1, label2)):
        if l1.argmax() != l2.argmax():
            num_mix = int(f2.size * ratio)
            indices = np.random.choice(f2.size, size=num_mix, replace=False)

            print(feat2[idx][:5])
            # 将选择的元素的值置零，并将数组重新调整回原始形状
            f2.flat[indices] = f1.flat[indices]
            print(f2[:5], feat2[idx][:5])
