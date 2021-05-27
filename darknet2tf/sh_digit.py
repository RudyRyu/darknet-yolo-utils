import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf

import utils
import common


STRIDES = np.array([16])
NUM_ANCHOR = 6
ANCHORS = np.array([7,7, 13,31, 21,45, 25,32, 28,48, 38,43]).reshape(1,6,2)
NUM_CLASS = 16
XYSCALE = [1]
INPUT_SIZE_HW = (64, 128)
OUTPUT_SIZES_HW = [(INPUT_SIZE_HW[0]//8, INPUT_SIZE_HW[1]//8)]
FILTER_MULTIPLIER = 1
SCORE_THRESH = 0.4
LAYER_SIZE = 10
OUTPUT_POS = [9]


def get_model_outputs(input_data):
    f = FILTER_MULTIPLIER

    x = common.convolutional(input_data, (3, 3, 3, 32*f))
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = common.convolutional(x, (3, 3, 32*f, 64*f))
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = common.convolutional(x, (3, 3, 64*f, 64*f))
    x = common.convolutional(x, (1, 1, 64*f, 32*f))
    x = common.convolutional(x, (3, 3, 32*f, 64*f))

    route1 = x
    route1 = common.reorg(route1, 2)

    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = common.convolutional(x, (3, 3, 64*f, 128*f))
    x = common.convolutional(x, (1, 1, 128*f, 64*f))
    x = common.convolutional(x, (3, 3, 64*f, 128*f))

    x = tf.concat([x, route1], axis=-1)
    x = common.convolutional(x, (3, 3, 384*f, 128*f))
    
    out1 = common.convolutional(
        x, (1, 1, 128*f, (NUM_CLASS+5)*NUM_ANCHOR),
        activate=False, bn=False)

    return [out1]

if __name__ == '__main__':

    params = {
        'get_model_func': get_model_outputs,
        'weights_file': 'cfg/digit/sh_digit_best.weights',
        'save_output_dir': 'sh_digit/checkpoint/',
        'layer_size': LAYER_SIZE,
        'output_pos': OUTPUT_POS,
        'input_size_wh': INPUT_SIZE_HW,
        'output_sizes_wh': OUTPUT_SIZES_HW,
        'num_class': NUM_CLASS,
        'anchors': ANCHORS,
        'num_anchor': NUM_ANCHOR,
        'strides': STRIDES,
        'xyscale': XYSCALE,
        'score_thresh': SCORE_THRESH
    }

    utils.save_tf(**params)
