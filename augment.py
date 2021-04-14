import json
import os
import random
import time
from collections import defaultdict
from pprint import pprint

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def transform():
    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=65, y1=100, x2=200, y2=150),
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

    cv2.imshow('before', image_before)
    cv2.imshow('after', image_after)
    cv2.waitKey()

def example(imgs, pieceWiseAffine, rotate):
    # 옵션별로 aug 설정
    if imgs.dtype != np.uint8:
        imgs = imgs.astype(np.uint8)

    if_ = lambda tf, t, f: t if tf else f

    ia.seed(int(time.time()))
    sometimes1 = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes3 = lambda aug: iaa.Sometimes(0.3, aug)
    sometimes5 = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes7 = lambda aug: iaa.Sometimes(0.7, aug)
    sometimes9 = lambda aug: iaa.Sometimes(0.9, aug)

    seq = iaa.Sequential([
        sometimes5(iaa.PiecewiseAffine(
                        scale=if_(pieceWiseAffine, (0.03, 0.03), 0.0))),

        sometimes1(iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)),
        sometimes3(iaa.Affine(shear=(-12,12))),
        iaa.Crop(percent=(0.0, 0.25)), # 항상 하는게 좋음 (crop2 기준)
        iaa.SomeOf(if_(rotate, 1, 0),
        [
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270)
        ]),

        sometimes3(iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5)),

        sometimes1(iaa.Dropout(p=(0, 0.01), per_channel=0.5)),
        sometimes1(iaa.Add((-40, 40), per_channel=0.5)),
        sometimes1(iaa.Sharpen(alpha=(0.3, 0.7), lightness=(0.75, 1.25))),
        sometimes1(iaa.MedianBlur(k=(3, 9))),
        sometimes1(iaa.GaussianBlur(sigma=(1.0, 2.5))),
        sometimes1(iaa.Grayscale(alpha=(0.1, 1.0))),

        sometimes1(iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((30, 70))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ])),

    ], random_order=False) # apply augmenters in random order

    aug_imgs = seq.augment_images(imgs)

    if aug_imgs.dtype != np.float32:
        aug_imgs = aug_imgs.astype(np.float32)

    return aug_imgs


if __name__ == '__main__':

    aug_imgs = example()
