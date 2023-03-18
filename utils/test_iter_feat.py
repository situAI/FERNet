import h5py
import numpy as np

wav2vec_file = h5py.File('/data1/ABAW/ABAW5/Aff-Wild2/feat/wav2vec_emotion.h5', 'r')

for video_name in wav2vec_file.keys():
    for frame_id in wav2vec_file[video_name].keys():
        feat = np.asarray(wav2vec_file[f'{video_name}/{frame_id}'])
        print(feat.shape)
