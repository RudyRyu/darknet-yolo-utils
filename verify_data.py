import glob
import os
import random
from pprint import pprint


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


if __name__=='__main__':
    new_train_valid_txt(
        train_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/train.txt',
        valid_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/valid.txt'
    )