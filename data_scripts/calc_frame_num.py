import os
import cv2


def main():
    video_path = '/data1/ABAW/ABAW5/Aff-Wild2/videos/videos'
    anno_path = '/data1/ABAW/ABAW5/Aff-Wild2/annotations'
    tmp_dirs = ['AU_Detection_Challenge/Train_Set', 'AU_Detection_Challenge/Validation_Set', 'EXPR_Classification_Challenge/Train_Set', 'EXPR_Classification_Challenge/Validation_Set', 'VA_Estimation_Challenge/Train_Set', 'VA_Estimation_Challenge/Validation_Set']
    frame_list = []
    for fn in os.listdir(video_path):
        bn = os.path.splitext(fn)[0]
        had_flag = 0
        for tmp_dir in tmp_dirs:
            txt_path = os.path.join(anno_path, tmp_dir, bn+'.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    line_num = len(f.read().splitlines())
                    frame_list.append([fn, 0, line_num])
                    had_flag = 1
                    break
        if not had_flag:
            cap = cv2.VideoCapture(os.path.join(video_path, fn))
            frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_list.append([fn, 1, int(frame_num)])
    with open('video_frame_num.txt', 'w') as f:
        for a, b, c in frame_list:
            f.write(f'{a}\t{b}\t{c}\n')


if __name__ == '__main__':
    main()

