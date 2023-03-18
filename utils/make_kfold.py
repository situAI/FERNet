import os
import json

root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/'
va_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/VA_Estimation_Challenge/'
expr_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/EXPR_Classification_Challenge/'
au_root = '/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/'
save_root = '/data1/ABAW/ABAW5/Aff-Wild2/kfold/'

def generate_kfold_json(anno_root, train_fold_num):
    fold_dict = {}
    print(f"task: {anno_root.split('/')[-2]}")
    train_root = os.path.join(anno_root, 'Train_Set')
    val_root = os.path.join(anno_root, 'Validation_Set')
    train_list = os.listdir(train_root)
    val_list = os.listdir(val_root)
    len_train_list = len(train_list)
    len_val_list = len(val_list)

    train_list = [p.replace('.txt', '') for p in train_list]
    val_list = [p.replace('.txt', '') for p in val_list]

    print(len_train_list)
    print(len_val_list)
    print(len_train_list / len_val_list)

    
    cnt = 0
    for i in range(0, len_train_list, train_fold_num):
        fold_dict[f'fold{cnt}'] = train_list[i: i + train_fold_num]
        cnt += 1
    fold_dict[f'fold{cnt + 1}'] = val_list

    print(fold_dict.keys())
    for fold_n in fold_dict.keys():
        print(f"{fold_n}: {len(fold_dict[fold_n])}")

    return fold_dict

va_kfold_dict = generate_kfold_json(va_root, 72)
expr_kfold_dict = generate_kfold_json(expr_root, 62)
# generate_kfold_json(au_root)

with open(os.path.join(save_root, 'va.json'), 'w') as f:
    json.dump(va_kfold_dict, f)


with open(os.path.join(save_root, 'expr.json'), 'w') as f:
    json.dump(expr_kfold_dict, f)
