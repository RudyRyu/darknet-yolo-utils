import json
import random
import os
from collections import defaultdict
from pprint import pprint

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split


def transform():
    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=65, y1=100, x2=200, y2=150),
        BoundingBox(x1=150, y1=80, x2=200, y2=130)
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
    