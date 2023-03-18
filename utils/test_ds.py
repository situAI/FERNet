import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm



class SequenceData(Dataset):
    def __init__(self, feat_root, label_root, feat_dict, seq_len, task, pad_mode='repeat_last'):
        self.feat_root = feat_root
        self.label_root = label_root
        self.feat_dict = feat_dict
        self.seq_len = seq_len
        self.task = task
        self.pad_mode = pad_mode
        self.feat_map = dict()
        self.sequence_list, self.label_list = self.make_sequence()

    def get_txt_contents(self, path):
        with open(path, 'r') as f:
            content = dict()
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                if self.task == 'va':
                    value_list = [float(value) for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list
                elif self.task == 'expr':
                    value_list = int(line.strip('\n'))
                    content[f'{i :05d}'] = value_list
                elif self.task == 'au':
                    value_list = [int(value) for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list

        return content


    def get_video_list(self):
        video_list = [x.split('.')[0] for x in sorted(os.listdir(self.label_root))]

        return video_list

    def __filter_invalid_annotations(self, label_dict, video_name):
        self.open_h5()

        returned_label_dict = label_dict.copy()
        if self.task == 'va':
            for seq_id in label_dict.keys():
                if (-5 in label_dict[seq_id]) or (f'{video_name}/{seq_id}' not in self.feat_map[list(self.feat_dict.keys())[0]].keys()):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'expr':
            for seq_id in label_dict.keys():
                if (-1 == label_dict[seq_id]) or (f'{video_name}/{seq_id}' not in self.feat_map[list(self.feat_dict.keys())[0]].keys()):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'au':
            for seq_id in label_dict.keys():
                if -1 in label_dict[seq_id]:
                    returned_label_dict.pop(seq_id)

        self.close_h5()

        return returned_label_dict


    def make_sequence_id_list(self, label_dict, video_name):
        label_dict = self.__filter_invalid_annotations(label_dict, video_name)
        sequence_id_list = list(label_dict.keys())
        sequence_label_list = list()
        if self.pad_mode == 'repeat_last':
            sequence_id_list = [sequence_id_list[i: i + self.seq_len] for i in range(0, len(sequence_id_list), self.seq_len)]

            for seq in sequence_id_list:
                for i in range(len(seq)):
                    seq[i] = video_name + '/' + str(seq[i])

            for i in range(len(sequence_id_list)):
                if len(sequence_id_list[i]) < self.seq_len:
                    pad_list = sequence_id_list[i]
                    while (len(pad_list) < self.seq_len):
                        pad_list.append(pad_list[-1])
                    sequence_id_list[i] = pad_list
            
            for sequence_id in sequence_id_list:
                sequence_label_list.append([label_dict[k.split('/')[-1]] for k in sequence_id])


            return sequence_id_list, sequence_label_list


    def make_sequence(self):
        seq_id_list = list()
        seq_label_list = list()
        video_list = self.get_video_list()
        for video in video_list:
            txt_path = os.path.join(self.label_root, video + '.txt')
            label_dict = self.get_txt_contents(txt_path)
            crt_id_list, crt_label_list = self.make_sequence_id_list(label_dict, video)
            seq_id_list += crt_id_list
            seq_label_list += crt_label_list

        return seq_id_list, seq_label_list


    def open_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name] = h5py.File(os.path.join(self.feat_root, feat_name + '.h5'), 'r')

    def close_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name].close()


    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        self.open_h5()
        seq_list = self.sequence_list[idx]
        lb_list = self.label_list[idx]
        assert len(seq_list) == self.seq_len == len(lb_list)
        seq_feat = list()
        seq_label = list()
        for seq_name, label in zip(seq_list, lb_list):
            feat = list()
            for feat_name in self.feat_map.keys():
                name = seq_name
                if feat_name in ['wav2vec', 'wav2vec_emotion', 'hubert', 'fbank']:
                    if '_right' in name:
                        name = name.replace('_right', '')
                    if '_left' in name:
                        name = name.replace('_left', '')

                    if name not in self.feat_map[feat_name].keys():
                        video_name = name.split('/')[0]
                        last_frame_name = list(self.feat_map[feat_name][video_name].keys())[-1]
                        crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{last_frame_name}'], dtype=np.float32)
                else:
                    crt_feat = np.asarray(self.feat_map[feat_name][name], dtype=np.float32)
                feat.append(crt_feat)
            feat = np.concatenate(feat, axis=-1)
            seq_feat.append(feat)
            seq_label.append(np.asarray(label))

        seq_feat = np.asarray(seq_feat, dtype=np.float32)
        seq_label = np.asarray(seq_label, dtype=np.float32)

        self.close_h5()

        return seq_feat, seq_label


feat_root = '/data1/ABAW/ABAW5/Aff-Wild2/feat/'
label_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/VA_Estimation_Challenge/Train_Set/'
# label_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/Validation_Set/' # check done
# label_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/EXPR_Classification_Challenge/Train_Set/' # check done
feat_dict = {'fau': 512, 'wav2vec': 1024}

def bcf(batch):
    feats, labels = list(), list()
    for crt_feat, crt_label in batch:
        feats.append(crt_feat)
        labels.append(crt_label)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    labels = torch.from_numpy(np.asarray(labels)).transpose(0, 1)

    return {'feat': feats, 'label': labels}



# ds = SequenceData(feat_root, label_root, feat_dict, 128, 'expr', 'repeat_last')
ds = SequenceData(feat_root, label_root, feat_dict, 128, 'va', 'repeat_last')
loader = DataLoader(ds, 32, False, collate_fn=bcf, num_workers=32, pin_memory=True)

for i, data in enumerate(tqdm(loader)):
    continue

