import os
import random

def split(input_total, out_train, out_val):
    f_in = open(input_total, "r")
    total = f_in.readlines()
    random.shuffle(total)

    train_num = int(len(total)*0.85)
    train = total[:train_num]
    val = total[train_num:]

    with open(out_train, 'w') as f:
        for line in train:
            f.write(line)

    with open(out_val, 'w') as f:
        for line in val:
            f.write(line)

split(
    input_total='/Users/rav/Desktop/plate_train/info/all_img_list_for_train.txt',
    out_train='/Users/rav/Desktop/plate_train/info/split_train_vs.txt',
    out_val='/Users/rav/Desktop/plate_train/info/split_valid_vs.txt')

