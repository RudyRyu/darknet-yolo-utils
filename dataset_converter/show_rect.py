import cv2
import os

def show(label_file):
    with open(label_file, "r") as f:
        for line in f.readlines():
            split = line.split()
            file_path = split[0]

            img = cv2.imread(file_path)

            for rect in split[1:]:
                l, t, r, b = list(map(int, rect.split(',')))[:4]

                cv2.rectangle(img, (l,t), (r,b), (0,255,0), 3)

            print(file_path)
            w = 512
            # h = int(img.shape[0] * (w/img.shape[1]))
            h = 384

            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LANCZOS4)
            print(img.shape)
            cv2.imshow('img', img)
            cv2.waitKey()

show('voc_output2.txt')
