import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/'

def cnt_va(root, video_list, phase):
    valence_list = list()
    arousal_list = list()

    for video in video_list:
        video_path = os.path.join(root, video)
        with open(video_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                value_list = [float(x) for x in line.strip('\n').split(',')]
                valence_list.append(value_list[0])
                arousal_list.append(value_list[1])

    valid_valence_list = list()
    valid_arousal_list = list()

    for (valence, arousal) in zip(valence_list, arousal_list):
        if valence != -5:
            valid_valence_list.append(valence)
            valid_arousal_list.append(arousal)

    print(f'va {phase} total videos: {len(video_list)}')
    print(f'va {phase} total frames: {len(valence_list)}')
    print(f'va {phase} valid frames: {len(valid_valence_list)}')
    print(f'va {phase} annotation %: {len(valid_valence_list) / len(valence_list)}')


    plt.figure(f'{phase}', figsize=(9, 9))
    plt.title(f'{phase}_va', color='blue')
    plt.scatter(valid_valence_list, valid_arousal_list)
    plt.draw()
    plt.savefig(f'{phase}_va.jpg')
    plt.close()


def cnt_expr(root, video_list, phase):
    expr_list = list()
    exp_tick_label = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']

    for video in video_list:
        video_path = os.path.join(root, video)
        with open(video_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                expr = int(line.strip('\n'))
                expr_list.append(expr)

    valid_expr_list = list()
    valid_expr_dict = dict()
    valid_expr_dict[0] = 0
    valid_expr_dict[1] = 0
    valid_expr_dict[2] = 0
    valid_expr_dict[3] = 0
    valid_expr_dict[4] = 0
    valid_expr_dict[5] = 0
    valid_expr_dict[6] = 0
    valid_expr_dict[7] = 0

    for expr in expr_list:
        if expr != -1:
            valid_expr_list.append(expr)
            valid_expr_dict[expr] += 1

    print(f'expr {phase} total videos: {len(video_list)}')
    print(f'expr {phase} total frames: {len(expr_list)}')
    print(f'expr {phase} valid frames: {len(valid_expr_list)}')
    print(f'expr {phase} annotation %: {len(valid_expr_list) / len(expr_list)}')
    print(f'expr valid dist: {valid_expr_dict}')

    for i, k in enumerate(valid_expr_dict.keys()):
        print(f'{exp_tick_label[i]}: {valid_expr_dict[k] / len(valid_expr_list)}')

    return valid_expr_dict

def cnt_au(root, video_list, phase):
    au_list = list()
    au_tick_label = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    for video in video_list:
        video_path = os.path.join(root, video)
        with open(video_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                au = [int(x) for x in line.strip('\n').split(',')]
                au_list.append(au)

    valid_au_list = list()
    valid_au_dict = dict()
    valid_au_dict[0] = 0
    valid_au_dict[1] = 0
    valid_au_dict[2] = 0
    valid_au_dict[3] = 0
    valid_au_dict[4] = 0
    valid_au_dict[5] = 0
    valid_au_dict[6] = 0
    valid_au_dict[7] = 0
    valid_au_dict[8] = 0
    valid_au_dict[9] = 0
    valid_au_dict[10] = 0
    valid_au_dict[11] = 0

    for au in au_list:
        if -1 in au:
            if 1 in au:
                print(au)
            else:
                continue
        valid_au_list.append(au)
        for i, k in enumerate(valid_au_dict.keys()):
            valid_au_dict[k] += au[i]


    print(f'au {phase} total videos: {len(video_list)}')
    print(f'au {phase} total frames: {len(au_list)}')
    print(f'au {phase} valid frames: {len(valid_au_list)}')
    print(f'au {phase} annotation %: {len(valid_au_list) / len(au_list)}')
    print(f'au valid dist: {valid_au_dict}')

    for i, k in enumerate(valid_au_dict.keys()):
        print(f'{au_tick_label[i]}: {valid_au_dict[k] / len(valid_au_list)}')

    return valid_au_dict



def va(path):
    train_root = os.path.join(root, 'VA_Estimation_Challenge', 'Train_Set')
    val_root = os.path.join(root, 'VA_Estimation_Challenge', 'Validation_Set')

    train_video_list = os.listdir(train_root)
    val_video_list = os.listdir(val_root)

    cnt_va(train_root, train_video_list, 'train')
    cnt_va(val_root, val_video_list, 'val')

def expr(path):
    train_root = os.path.join(root, 'EXPR_Classification_Challenge', 'Train_Set')
    val_root = os.path.join(root, 'EXPR_Classification_Challenge', 'Validation_Set')

    train_video_list = os.listdir(train_root)
    val_video_list = os.listdir(val_root)

    train_valid_expr_dict = cnt_expr(train_root, train_video_list, 'train')
    val_valid_expr_dict = cnt_expr(val_root, val_video_list, 'val')


    exp_x = np.arange(8)
    train_exp_list = list()
    val_exp_list = list()
    for k in train_valid_expr_dict.keys():
        train_exp_list.append(train_valid_expr_dict[k])
        val_exp_list.append(val_valid_expr_dict[k])
    bar_width = 0.3
    exp_tick_label = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
    plt.figure('exp', figsize=(9, 9))
    plt.bar(exp_x, train_exp_list, bar_width, color='salmon', label='train')
    plt.bar(exp_x + bar_width, val_exp_list, bar_width, color='orchid', label='val')
    plt.legend()
    plt.xticks(exp_x + bar_width / 2, exp_tick_label)
    plt.savefig('./expr.jpg')

def au(path):
    train_root = os.path.join(root, 'AU_Detection_Challenge', 'Train_Set')
    val_root = os.path.join(root, 'AU_Detection_Challenge', 'Validation_Set')

    train_video_list = os.listdir(train_root)
    val_video_list = os.listdir(val_root)

    train_valid_au_dict = cnt_au(train_root, train_video_list, 'train')
    val_valid_au_dict = cnt_au(val_root, val_video_list, 'val')

    au_x = np.arange(12)
    train_au_list = list()
    val_au_list = list()
    for k in train_valid_au_dict.keys():
        train_au_list.append(train_valid_au_dict[k])
        val_au_list.append(val_valid_au_dict[k])
    bar_width = 0.3
    au_tick_label = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    plt.figure('au', figsize=(9, 9))
    plt.bar(au_x, train_au_list, bar_width, color='salmon', label='train')
    plt.bar(au_x + bar_width, val_au_list, bar_width, color='orchid', label='val')
    plt.legend()
    plt.xticks(au_x + bar_width / 2, au_tick_label)
    plt.savefig('au.jpg')



# va(root)
# expr(root)
au(root)
