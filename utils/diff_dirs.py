import os

dir1 = '/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned/'
dir2 = '/data1/ABAW/ABAW5/Aff-Wild2/cropped_aligned_images/cropped_aligned_new_50_vids/'

dir1_list = os.listdir(dir1)
dir2_list = os.listdir(dir2)

union_list = list(set(dir1_list) & set(dir2_list))

print(union_list)
