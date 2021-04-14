
import os
import json
from pprint import pprint

import cv2


def convert_voc_to_yolo_org(voc_txt, img_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_in = open(voc_txt, "r")

    for line in f_in.readlines():
        split = line.split()

        img_path = os.path.join(img_dir, split[0])
        try:
            img = cv2.imread(img_path)
            file_name = os.path.splitext(os.path.basename(split[0]))[0]

            f_out = open(os.path.join(output_dir, file_name) + '.txt', 'w+')
            for infos in split[1:]:
                l, t, r, b, lbl = list(map(int, infos.split(',')))

                w, h = r-l, b-t
                c_x, c_y = l+w/2, t+h/2
                img_h, img_w = img.shape[:2]

                w, h = w/img_w, h/img_h
                c_x, c_y = c_x/img_w, c_y/img_h

                f_out.write(f'{lbl} {c_x} {c_y} {w} {h}\n')

            f_out.close()

        except :
            print(split)
            raise()

    f_in.close()

convert_voc_to_yolo_org(
    voc_txt='/Users/rav/Desktop/plate_train/dataset/netvision/netvision_voc_output.txt',
    img_dir='/Users/rav/Desktop/plate_train/dataset/netvision',
    output_dir='/Users/rav/Desktop/plate_train/dataset/netvision')
