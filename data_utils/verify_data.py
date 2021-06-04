import glob
import multiprocessing as mp
import os
import random
from functools import partial
from pprint import pprint

import cv2


def new_train_valid_txt(train_path, valid_path):
    train = open(train_path, 'r')
    train_lines = train.readlines()

    new_train = open('new_train.txt', 'w')
    for path in train_lines:
        if os.path.exists(path.strip()):
            new_train.write(path)

    new_valid = open('new_valid.txt', 'w')
    valid = open(valid_path, 'r')
    valid_lines = valid.readlines()
    for path in valid_lines:
        if os.path.exists(path.strip()):
            new_valid.write(path)


def read_data_and_rewrite_txt(data_dir, output_dir, valid_ratio=0.2):
    total_list = []

    for image_path in glob.glob(os.path.join(data_dir, '*.jpg')):
        total_list.append(image_path)

    total_list.sort()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'total.txt'), 'w') as f:
        for line in total_list:
            f.write(line+'\n')
            
    random.shuffle(total_list)
    valid_num = int(len(total_list)*valid_ratio)
    valid_list = total_list[:valid_num]
    train_list = total_list[valid_num:]

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for line in train_list:
            f.write(line+'\n')

    with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
        for line in valid_list:
            f.write(line+'\n')


def _create_padding_image_and_label_worker(image_name, ratio_wh, output_dir):

    image_path = image_name + '.jpg'
    txt_path = image_name + '.txt'
    image = cv2.imread(image_path)

    ratio = ratio_wh[0] / ratio_wh[1]
    img_h, img_w = image.shape[:2]

    f_label = open(txt_path, 'r')
    padding_label = ''
    if img_w / img_h < ratio:
        target_w = img_h * ratio
        pad_w = target_w - img_w
        l_pad = round(random.uniform(0, 1) * pad_w)
        r_pad = round(pad_w - l_pad)
        padding_image = cv2.copyMakeBorder(image,
            0, 0, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        pad_w = padding_image.shape[1]

        for label in f_label:
            class_id, x, y, w, h = label.split()

            x = float(x) * img_w
            w = float(w) * img_w

            x = (l_pad+x) / pad_w
            w = w / pad_w
            padding_label += f'{class_id} {x} {y} {w} {h}\n'

    elif img_w / img_h > ratio:
        target_h = img_w / ratio
        pad_h = target_h - img_h
        t_pad = round(random.uniform(0, 1) * pad_h)
        b_pad = round(pad_h - t_pad)
        padding_image= cv2.copyMakeBorder(image,
            t_pad, b_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        pad_h = padding_image.shape[0]

        for label in f_label:
            class_id, x, y, w, h = label.split()

            y = float(y) * img_h
            h = float(h) * img_h

            y = (t_pad+y) / pad_h
            h = h / pad_h

            padding_label += f'{class_id} {x} {y} {w} {h}\n'

    else:
        padding_image = image
        for label in f_label:
            class_id, x, y, w, h = label.split()
            padding_label += f'{class_id} {x} {y} {w} {h}\n'

    image_name_path = os.path.join(output_dir, os.path.basename(image_name))

    cv2.imwrite(image_name_path+'.jpg', padding_image)

    with open(image_name_path+'.txt', 'w') as f:
        f.write(padding_label)

    # labels = padding_label.split('\n')
    # pad_h, pad_w = padding_image.shape[:2]
    # for label in labels[:-1]:
    #     class_id, x, y, w, h = label.split()
    #     x = float(x) * pad_w
    #     y = float(y) * pad_h
    #     w = float(w) * pad_w
    #     h = float(h) * pad_h

    #     x = round(x - (w/2))
    #     y = round(y - (h/2))
    #     w = round(w)
    #     h = round(h)


def add_padding_multiprocess(data_dir, ratio_wh, output_dir, 
                             process_num=mp.cpu_count()):

    image_names = []
    for image_path in sorted(glob.glob(os.path.join(data_dir, '*.jpg'))):
        image_name = os.path.splitext(image_path)[0]
        image_names.append(image_name)

    os.makedirs(output_dir, exist_ok=True)

    partial_map = partial(_create_padding_image_and_label_worker, 
        ratio_wh=ratio_wh,
        output_dir=output_dir
    )

    pool = mp.Pool(processes=process_num)
    pool.map(partial_map, image_names)

        
if __name__=='__main__':
    # new_train_valid_txt(
    #     train_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/train.txt',
    #     valid_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/valid.txt'
    # )

    # read_data_and_rewrite_txt(
    #     data_dir='/Users/rudy/Desktop/untitled folder 2/data/',
    #     output_dir='/Users/rudy/Desktop/untitled folder 2/output/')

    add_padding_multiprocess(
        data_dir='/Users/rudy/Desktop/digit_data_noblur/data',
        ratio_wh=(2,1),
        output_dir='/Users/rudy/Desktop/digit_data_noblur/padded_data'
    )