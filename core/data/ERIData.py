import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os

from utils.registery import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ERIData(Dataset):
    def __init__(self,
                 feat_root,
                 label_root,
                 feat_dict,
                 seq_len,
                 task,
                 pad_mode='repeat_last',
                 mask_aug=False,
                 mask_ratio=0.0,
                 label_smooth=False,
                 smooth_factor=0.0):

        """ERIData

        Args:
            feat_root (str): feat root path
            label_root (str): label root path
            feat_dict (dict): feat dict in which key is `feat_name`, value is `feat_dim`
            seq_len (int): sequence length
            task (str): `va`, `expr` or `au`
            pad_mode (str): pad mode, here just implemented `repeat_last` (default: 'repeat_last')
        """
        self.feat_root = feat_root
        self.label_root = label_root
        self.feat_dict = feat_dict
        self.seq_len = seq_len
        self.task = task
        self.pad_mode = pad_mode
        self.feat_map = dict()
        self.video_label_dict, self.video_seq_len_dict, self.id_video_dict = self.make_sequence()
        self.mask_aug = mask_aug
        self.mask_ratio = mask_ratio
        self.label_smooth = label_smooth
        self.smooth_factor = smooth_factor

    def make_sequence(self):
        """make sequence

        Returns:
            seq_id_list (list): explained in :method `make_sequence_id_list`
            seq_label_list (list): explained in :method `make_sequence_id_list`
        """
        txt_path = self.label_root
        video_label_dict = {}
        video_seq_len_dict = {}
        id_video_dict = {}
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
            for idx, line in enumerate(lines):
                video_items = line.split(' ')
                video_name = video_items[0]
                seq_len = video_items[1]
                label = video_items[2:]
                video_label_dict[video_name] = [float(i) for i in label]
                video_seq_len_dict[video_name] = seq_len
                id_video_dict[idx] = video_name

        return video_label_dict, video_seq_len_dict, id_video_dict


    def open_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name] = h5py.File(os.path.join(self.feat_root, feat_name + '.h5'), 'r')

    def close_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name].close()

    def __len__(self):
        return len(self.id_video_dict)

    def __getitem__(self, idx):
        try:
            self.open_h5()
            video_name = self.id_video_dict[idx]
            crt_seq_len = int(self.video_seq_len_dict[video_name])
            seq_label = self.video_label_dict[video_name]
            seq_feat = list()
            # s = random.randint(1, int(crt_seq_len*0.2))
            # e = random.randint(int(crt_seq_len*0.8), crt_seq_len)
            # crt_seq_len = e - s 
            stride = crt_seq_len / self.seq_len
	
            frame_idxs = [int(i*stride+1) for i in range(0, self.seq_len)]
            for frame_id in frame_idxs:
                feat = list()
                for feat_name in self.feat_map.keys():
                    if 'ecapatdnn_train_val' == feat_name:
                        continue
                    crt_feat = None
                    # if self.feat_map[feat_name][video_name].get(f'{frame_id:05d}') is not None:
                    try:
                        crt_feat = np.asarray(self.feat_map[feat_name][video_name][f'{frame_id:05d}'])
                    except:
                        for crt_id in range(1, crt_seq_len):
                            crt_id_upper = int(frame_id) + crt_id
                            crt_id_lower = int(frame_id) - crt_id
                            upper_key = f'{crt_id_upper :05d}'
                            lower_key = f'{crt_id_lower :05d}'
                            if self.feat_map[feat_name][video_name].get(upper_key) is not None:
                                crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{upper_key}'])
                                break
                            elif self.feat_map[feat_name][video_name].get(lower_key) is not None:
                                crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{lower_key}'])
                                break
                    assert crt_feat is not None, f'crt_feat is empty, feat_name: {feat_name}, frame_id: {frame_id}'
                    feat.append(crt_feat)
                feat = np.concatenate(feat, axis=-1)
                seq_feat.append(feat)

            seq_feat = np.asarray(seq_feat)
            seq_label = np.asarray(seq_label)

            if self.label_smooth:
                # 计算标签平滑后的目标
                seq_label = seq_label * (1 - self.smooth_factor) + self.smooth_factor / seq_label.shape[1]

            if self.mask_aug:
                seq_feat = random_mask(seq_feat, self.mask_ratio)

            self.close_h5()

            return seq_feat, seq_label,  video_name
        except:
            # print(f'wrong: {self.id_video_dict[idx]}')
            return self.__getitem__(random.randint(0, len(self)-1))


def random_mask(arr, ratio=0.0):
    arr_shape = arr.shape
    if random.random() < 0.5:
        # 将数组转化为一维数组，并随机选择ratio%的元素的索引
        num_zeros = int(arr.size * ratio)
        zero_indices = np.random.choice(arr.size, size=num_zeros, replace=False)

        # 将选择的元素的值置零，并将数组重新调整回原始形状
        arr = arr.flatten()
        arr[zero_indices] = 0
        arr = arr.reshape(arr_shape)

    return arr
