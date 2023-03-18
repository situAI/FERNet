import os
import h5py

va_anno_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/VA_Estimation_Challenge/'
expr_anno_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/EXPR_Classification_Challenge/'
au_anno_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/'

feat_root = '/data1/ABAW/ABAW5/Aff-Wild2/feat/'
fau_h5 = h5py.File(os.path.join(feat_root, 'fau.h5'), 'r')

def get_txt_num(path, task):
    valid_num = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if task == 'va':
                value = [float(x) for x in line.strip('\n').split(',')]
                if -5 not in value:
                    valid_num += 1
            elif task == 'expr':
                value = int(line.strip('\n'))
                if -1 != value:
                    valid_num += 1
            elif task == 'au':
                value = [int(x) for x in line.strip('\n').split(',')]
                if -1 not in value:
                    valid_num += 1

    return valid_num


def get_cropalign_num(h5_file, video_name):
    return len(list(h5_file[video_name].keys()))


def sta(root, h5_file, task):
    crop_align_num = 0
    txt_num = 0
    txt_file_list = os.listdir(root)
    for txt_file in txt_file_list:
        video_name = txt_file.split('.')[0]
        crop_align_num += get_cropalign_num(h5_file, video_name)
        txt_num += get_txt_num(os.path.join(root, txt_file), task)

    return crop_align_num, txt_num


def stats(root, h5_file, task):
    train_root = os.path.join(root, 'Train_Set')
    val_root = os.path.join(root, 'Validation_Set')

    train_crop_align_num, train_annotation_num = sta(train_root, h5_file, task)
    val_crop_align_num, val_annotation_num = sta(val_root, h5_file, task)

    print(f'task {task}: train crop_align num: {train_crop_align_num}, annotation num: {train_annotation_num}, diff num: {train_annotation_num - train_crop_align_num}')
    print(f'task {task}: val crop_align num: {val_crop_align_num}, annotation num: {val_annotation_num}, diff num: {val_annotation_num - val_crop_align_num}')

stats(va_anno_root, fau_h5, 'va')
stats(expr_anno_root, fau_h5, 'expr')
stats(au_anno_root, fau_h5, 'au')

fau_h5.close()
