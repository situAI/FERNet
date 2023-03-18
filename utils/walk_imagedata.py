import os

data_root = '/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned'

for root, dirs, files in os.walk(data_root):
    for name in files:
        print(os.path.join(root, name))

