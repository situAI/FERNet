import h5py
import os
import numpy as np
from tqdm import tqdm

# npy root path
npy_root = '/data1/ABAW/ABAW5/Aff-Wild2/feat/ecapatdnn/'

# feat root path
feat_root = '/data1/ABAW/ABAW5/Aff-Wild2/feat/'

#FIXME: feat name
h5_file = h5py.File(os.path.join(feat_root, 'ecapatdnn.h5'), 'w')

video_npy_names = os.listdir(npy_root)

for video_npy_name in tqdm(video_npy_names):
    frame_num = len(os.listdir(os.path.join(npy_root, video_npy_name)))
    for i in range(frame_num):
        frame_npy_path = os.path.join(npy_root, video_npy_name, f'{i :05d}.npy')
        frame_npy = np.load(frame_npy_path)
        crt_key = f'{video_npy_name}/{i :05d}'
        h5_file.create_dataset(f"{crt_key}", data=frame_npy, dtype='f')

h5_file.close()

