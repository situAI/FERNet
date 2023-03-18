import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transform
from mobilefacenet import MobileFaceNet


class ABAW5(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.img_list = list()
        for root, dirs, files in os.walk(data_root):
            for name in files:
                if name[-1] == 'g':
                    self.img_list.append(os.path.join(root, name))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        video_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]
        name = video_name + '/' + img_name
        img = Image.open(img_path)
        data = self.transform(img)

        return data, name

def get_abaw5_loader():
    root = '/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned/'
    tfm = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    ds = ABAW5(data_root=root, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=16, drop_last=False)

    return dl


def load_model():
    model = MobileFaceNet(embedding_size=512)
    model.load_state_dict(torch.load('./ckpt_epoch_5.pt', map_location='cpu'))
    model.cuda()
    model.eval()

    return model


@torch.no_grad()
def extract_h5(model, loader, phase_name):
    h5_root = '/data1/ABAW/ABAW5/Aff-Wild2/feat/'
    h5_file = h5py.File(os.path.join(h5_root, 'ms_va.h5'), 'w')
    print(f'processing {phase_name}...')
    for _, (img, name) in enumerate(tqdm(loader)):
        img = img.cuda()
        _, _, _, feat = model(img)
        feat = feat.detach().cpu().numpy()
        for i in range(feat.shape[0]):
            h5_file.create_dataset(f"{name[i]}", data=feat[i], dtype='f')

    h5_file.close()

def main():
    model = load_model()

    loader = get_abaw5_loader()
    extract_h5(model, loader, 'overall')

if __name__ == '__main__':
    main()
