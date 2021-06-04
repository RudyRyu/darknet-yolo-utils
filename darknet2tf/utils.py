import colorsys
import random

import cv2
import numpy as np
import tensorflow as tf


def decode_tf(conv_output, output_size_wh, 
              num_class, anchors, num_anchor, strides, xyscale,
              i=0):

    batch_size = tf.shape(conv_output)[0]

    conv_output = tf.reshape(conv_output,
        (batch_size, *output_size_wh, num_anchor, num_class+5))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = \
        tf.split(conv_output, (2, 2, 1, num_class), axis=-1)

    xy_grid = tf.meshgrid(
        tf.range(output_size_wh[1]), tf.range(output_size_wh[0]))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(
        tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, num_anchor, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy)*xyscale[i]) \
                - 0.5*(xyscale[i]-1)+xy_grid) * strides[i]

    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_class))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob


def filter_boxes(box_xywh, scores, 
                 score_threshold=0.4, 
                 input_shape=tf.constant([128,64])):

    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def load_weights(model, weights_file, layer_size, output_pos):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):

        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)

        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def get_tf_model(input_layer, feature_maps, input_size_wh, output_sizes_wh,
                 num_class, anchors, num_anchor, strides, xyscale,
                 score_thresh):
    
    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox, prob = decode_tf(fm, output_sizes_wh[i], 
            num_class, anchors, num_anchor, strides, xyscale)

        bbox_tensors.append(bbox)
        prob_tensors.append(prob)

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, 
        score_threshold=score_thresh, 
        input_shape=tf.constant(input_size_wh))

    pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)

    return model

def save_tf(get_model_func, weights_file, save_output_dir, 
            layer_size, output_pos,
            input_size_hw, output_sizes_hw,
            num_class, anchors, num_anchor, strides, xyscale,
            score_thresh):

    input_layer = tf.keras.layers.Input([*input_size_hw, 3])
    feature_maps = get_model_func(input_layer)

    model = get_tf_model(
        input_layer, feature_maps, input_size_hw, output_sizes_hw, 
        num_class, anchors, num_anchor, strides, xyscale,
        score_thresh)

    load_weights(model, weights_file, layer_size, output_pos)
    model.summary()
    model.save(save_output_dir)


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for class_id, name in enumerate(data):
            names[class_id] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: 
            continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    
    return image
