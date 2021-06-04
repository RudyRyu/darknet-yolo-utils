import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pprint import pprint
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

import sh_digit
import utils


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', 'sh_digit/checkpoint', 'path to weights')
flags.DEFINE_list('image_size_wh', [128,64], 'resize images to')
flags.DEFINE_string('model', 'sh_digit', 'model name')
flags.DEFINE_string('image_path', 'sample/test_images/test_image38.png', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.4, 'score threshold')


def main(_argv):
    config = ConfigProto()
    # config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    # model_module = sh_digit
    # if FLAGS.model == 'sh_digit':
    #     model_module = sh_digit

    # STRIDES = model_module.STRIDES
    # ANCHORS = model_module.ANCHORS
    # NUM_CLASS = model_module.NUM_CLASS
    # XYSCALE = model_module.XYSCALE

    original_image = cv2.imread(FLAGS.image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, tuple(FLAGS.image_size_wh))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    print(boxes)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    print(boxes.numpy())
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    image = utils.draw_bbox(original_image, pred_bbox,
        classes=utils.read_class_names('cfg/digit/digit.names'))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('result', image)
    cv2.waitKey()
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    # image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

