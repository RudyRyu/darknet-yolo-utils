import hashlib
import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pprint import pprint
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import cv2
# import tensorflow as tf

import augment

class_label_id_map = {
    '1' : 0,
    '2' : 1,
    '3' : 2,
    '4' : 3,
    '5' : 4,
    '6' : 5,
    '7' : 6,
    '8' : 7,
    '9' : 8,
    '0' : 9,
    '℃' : 10,
    '℉' : 11,
    ':' : 12,
    '.' : 13,
    '%' : 14,
    '-' : 15,
    'panel' : 16,
}

# g_image_feature_map = {
#     'image/height': tf.io.FixedLenFeature([], tf.int64),
#     'image/width': tf.io.FixedLenFeature([], tf.int64),
    
#     # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
#     'image/encoded': tf.io.FixedLenFeature([], tf.string),
#     # 'image/format': tf.io.FixedLenFeature([], tf.string),
    
#     'image/panel/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
#     'image/panel/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
#     'image/panel/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
#     'image/panel/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
    
#     'image/object/bbox/xmins': tf.io.VarLenFeature(tf.float32),
#     'image/object/bbox/xmaxs': tf.io.VarLenFeature(tf.float32),
#     'image/object/bbox/ymins': tf.io.VarLenFeature(tf.float32),
#     'image/object/bbox/ymaxs': tf.io.VarLenFeature(tf.float32),
#     'image/object/class/texts': tf.io.VarLenFeature(tf.string),
#     'image/object/class/ids': tf.io.VarLenFeature(tf.int64),
# }


# def build_tf_train_example(image_path, 
#                            width, height, 
#                            panel_ltrb,
#                            digits):

#     """
#     panel_ltrb:
#         - [xmin, ymin, xmax, ymax]
#         - e.g. [0.01, 0.05, 0.23, 0.15]
#     digits:
#         - [digit_dict1, digit_dict2, ...]
#         - e.g. digits[0]['class_text'] == '8'
#     """
#     img_raw = open(image_path, 'rb').read()
#     key = hashlib.sha256(img_raw).hexdigest()

#     xmins = []
#     ymins = []
#     xmaxs = []
#     ymaxs = []
#     classes_text = []
#     classes_id = []
#     for digit in digits:
#         xmins.append(digit['xmin'])
#         ymins.append(digit['ymin'])
#         xmaxs.append(digit['xmax'])
#         ymaxs.append(digit['ymax'])
#         classes_text.append(digit['class_text'].encode('utf8'))
#         classes_id.append(digit['class_id'])

#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image/height': tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[height])),
#         'image/width': tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[width])),
        
#         'image/key/sha256': tf.train.Feature(
#             bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
#         'image/encoded': tf.train.Feature(
#             bytes_list=tf.train.BytesList(value=[img_raw])),
#         'image/format': tf.train.Feature(
#             bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        
#         'image/panel/bbox/xmin': tf.train.Feature(
#             float_list=tf.train.FloatList(value=[panel_ltrb[0]])),
#         'image/panel/bbox/ymin': tf.train.Feature(
#             float_list=tf.train.FloatList(value=[panel_ltrb[1]])),
#         'image/panel/bbox/xmax': tf.train.Feature(
#             float_list=tf.train.FloatList(value=[panel_ltrb[2]])),
#         'image/panel/bbox/ymax': tf.train.Feature(
#             float_list=tf.train.FloatList(value=[panel_ltrb[3]])),
        
#         'image/object/bbox/xmin': tf.train.Feature(
#             float_list=tf.train.FloatList(value=xmins)),
#         'image/object/bbox/xmax': tf.train.Feature(
#             float_list=tf.train.FloatList(value=xmaxs)),
#         'image/object/bbox/ymin': tf.train.Feature(
#             float_list=tf.train.FloatList(value=ymins)),
#         'image/object/bbox/ymax': tf.train.Feature(
#             float_list=tf.train.FloatList(value=ymaxs)),
#         'image/object/class/text': tf.train.Feature(
#             bytes_list=tf.train.BytesList(value=classes_text)),
#         'image/object/class/id': tf.train.Feature(
#             int64_list=tf.train.Int64List(value=classes_id))
#     }))

#     return example


# def generate_tf_records_from_vott(vott_json, image_dir, output_dir):
#     with open(vott_json) as vott_buffer:
#         vott = json.loads(vott_buffer.read())

#     for i, v in enumerate(list(vott['assets'].values())[::-1]):
#         image_path = os.path.join(image_dir, v['asset']['name'])
#         image_h = v['asset']['size']['height'] 
#         image_w = v['asset']['size']['width']
#         # find panel
#         panels = []
#         for region in v['regions']:
#             if region['tags'][0] != 'panel':
#                 continue
                
#             h = float(region['boundingBox']['height'])
#             w = float(region['boundingBox']['width'])
#             l = float(region['boundingBox']['left'])
#             t = float(region['boundingBox']['top'])

#             r = l + w
#             b = t + h

#             panel_xmin = l / image_w
#             panel_ymin = t / image_h
#             panel_xmax = r / image_w
#             panel_ymax = b / image_h

#             panels.append((panel_xmin, panel_ymin, panel_xmax, panel_ymax))

#         # find elemets in each panel
#         for p, panel_ltrb in enumerate(panels):
#             digits = []
#             for region in v['regions']:
#                 if region['tags'][0] == 'panel':
#                     continue
                
#                 h = float(region['boundingBox']['height'])
#                 w = float(region['boundingBox']['width'])
#                 l = float(region['boundingBox']['left'])
#                 t = float(region['boundingBox']['top'])

#                 r = l + w
#                 b = t + h
                
#                 l /= image_w
#                 t /= image_h
#                 r /= image_w
#                 b /= image_h

#                 if all([l >= panel_ltrb[0],
#                         t >= panel_ltrb[1],
#                         r <= panel_ltrb[2],
#                         b <= panel_ltrb[3]]):
                    
#                     digit_dict = {}
#                     xmin = (l-panel_ltrb[0]) / (panel_ltrb[2]-panel_ltrb[0])
#                     ymin = (t-panel_ltrb[1]) / (panel_ltrb[3]-panel_ltrb[1])
#                     xmax = (r-panel_ltrb[0]) / (panel_ltrb[2]-panel_ltrb[0])
#                     ymax = (b-panel_ltrb[1]) / (panel_ltrb[3]-panel_ltrb[1])

#                     digit_dict['xmin'] = xmin
#                     digit_dict['ymin'] = ymin
#                     digit_dict['xmax'] = xmax
#                     digit_dict['ymax'] = ymax
#                     digit_dict['class_text'] = region['tags'][0]
#                     digit_dict['class_id'] = class_label_id_map[region['tags'][0]]
                    
#                     digits.append(digit_dict)

#             example = build_tf_train_example(
#                 image_path, image_w, image_h,
#                 panel_ltrb, digits
#             )

#             tfrecord_path = os.path.join(output_dir, f'{i:04d}_{p}.tfrecord')
#             with tf.io.TFRecordWriter(tfrecord_path) as writer:
#                 writer.write(example.SerializeToString())


def generate_yolo_org_from_vott(vott_json, image_dir, output_dir,
                                random_color=False,
                                transform_num_per_panel=5, valid_ratio=0.2):
    with open(vott_json) as vott_buffer:
        vott = json.loads(vott_buffer.read())

    data_output_dir = os.path.join(output_dir, 'data/')
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    total_list = []
    for v in list(vott['assets'].values())[::-1]:
        image_path = os.path.join(image_dir, v['asset']['name'])
        print(image_path)
        image_name = os.path.splitext(v['asset']['name'])[0]

        if image_name in ['0166_4', '0183_5', '0183_6', '0356_2', '0356_3']:
            continue
        
        image = cv2.imread(image_path)

        # find panel
        panels = []
        for region in v['regions']:
            if region['tags'][0] != 'panel':
                continue
                
            h = float(region['boundingBox']['height'])
            w = float(region['boundingBox']['width'])
            l = float(region['boundingBox']['left'])
            t = float(region['boundingBox']['top'])

            r = l + w
            b = t + h

            l,t,r,b = int(l), int(t), int(r), int(b)
            panels.append((l, t, r, b))

        # find elemets in each panel
        for p, panel_ltrb in enumerate(panels):
            digits = []
            
            for region in v['regions']:
                if region['tags'][0] == 'panel':
                    continue
                
                class_id = class_label_id_map[region['tags'][0]]

                if class_id in [10, 11, 14]:
                    continue

                h = float(region['boundingBox']['height'])
                w = float(region['boundingBox']['width'])
                l = float(region['boundingBox']['left'])
                t = float(region['boundingBox']['top'])

                r = l + w
                b = t + h
                
                if all([l >= panel_ltrb[0],
                        t >= panel_ltrb[1],
                        r <= panel_ltrb[2],
                        b <= panel_ltrb[3]]):

                    digit_dict = {}
                    digit_dict['class_id'] = class_id
                    digit_dict['l'] = l
                    digit_dict['t'] = t
                    digit_dict['r'] = r
                    digit_dict['b'] = b
                    
                    digits.append(digit_dict)

            for t in range(2):
                # not augmentation
                if t == 0:
                    # b means batch
                    b_images = np.expand_dims(image, axis=0)
                    b_panel, b_digits = [panel_ltrb], [digits]

                # augmentation
                else:
                    b_images, b_panel, b_digits = \
                        augment.random_augmentation(
                            image.copy(),
                            panel_ltrb, digits,
                            batch=transform_num_per_panel,
                            random_geometry=True,
                            random_color=random_color)

                for b, (aug_image, panel, digits) in enumerate(zip(b_images, 
                                                                   b_panel,
                                                                   b_digits)):

                    if t==0:
                        file_name = f'{image_name}_p{p}'
                    else:
                        file_name = f'{image_name}_p{p}_a{b}'

                    panel_image = aug_image[int(panel[1]):int(panel[3]),
                                            int(panel[0]):int(panel[2])]

                    f_out = open(
                        os.path.join(data_output_dir, file_name)+'.txt', 'w+')

                    total_list.append(
                        os.path.join('data/', f'{file_name}.jpg'))

                    cv2.imwrite(
                        os.path.join(data_output_dir, f'{file_name}.jpg'), 
                        panel_image)

                    for d in digits:
                        class_id = d['class_id']
                        l,t,r,b = d['l'], d['t'], d['r'], d['b']

                        xmin = (l-panel[0]) / (panel[2]-panel[0])
                        ymin = (t-panel[1]) / (panel[3]-panel[1])
                        xmax = (r-panel[0]) / (panel[2]-panel[0])
                        ymax = (b-panel[1]) / (panel[3]-panel[1])
                            
                        c_x = (xmin+xmax)/2
                        c_y = (ymin+ymax)/2
                        w = xmax-xmin
                        h = ymax-ymin

                        f_out.write(f'{class_id} {c_x} {c_y} {w} {h}\n')
                
                    f_out.close()


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




def _generate_worker(v, image_dir, data_output_dir, random_color, 
                     transform_num_per_panel, lock, total_list):

    image_path = os.path.join(image_dir, v['asset']['name'])
    image_name = os.path.splitext(v['asset']['name'])[0]

    if image_name in ['0166_4', '0183_5', '0183_6', '0356_2', '0356_3']:
        return
    
    image = cv2.imread(image_path)

    # find panel
    panels = []
    for region in v['regions']:
        if region['tags'][0] != 'panel':
            continue
            
        h = float(region['boundingBox']['height'])
        w = float(region['boundingBox']['width'])
        l = float(region['boundingBox']['left'])
        t = float(region['boundingBox']['top'])

        r = l + w
        b = t + h

        l,t,r,b = int(l), int(t), int(r), int(b)
        panels.append((l, t, r, b))

    # find elemets in each panel
    for p, panel_ltrb in enumerate(panels):
        digits = []
        
        for region in v['regions']:
            if region['tags'][0] == 'panel':
                continue
            
            class_id = class_label_id_map[region['tags'][0]]

            if class_id in [10, 11, 14]:
                continue

            h = float(region['boundingBox']['height'])
            w = float(region['boundingBox']['width'])
            l = float(region['boundingBox']['left'])
            t = float(region['boundingBox']['top'])

            r = l + w
            b = t + h
            
            if all([l >= panel_ltrb[0],
                    t >= panel_ltrb[1],
                    r <= panel_ltrb[2],
                    b <= panel_ltrb[3]]):

                digit_dict = {}
                digit_dict['class_id'] = class_id
                digit_dict['l'] = l
                digit_dict['t'] = t
                digit_dict['r'] = r
                digit_dict['b'] = b
                
                digits.append(digit_dict)

        for t in range(2):
            # not augmentation
            if t == 0:
                # b means batch
                b_images = np.expand_dims(image, axis=0)
                b_panel, b_digits = [panel_ltrb], [digits]

            # augmentation
            else:
                b_images, b_panel, b_digits = \
                    augment.random_augmentation(
                        image.copy(),
                        panel_ltrb, digits,
                        batch=transform_num_per_panel,
                        random_geometry=True,
                        random_color=random_color)

            for b, (aug_image, panel, digits) in enumerate(zip(b_images, 
                                                                b_panel,
                                                                b_digits)):

                if t==0:
                    file_name = f'{image_name}_p{p}'
                else:
                    file_name = f'{image_name}_p{p}_a{b}'

                panel_image = aug_image[int(panel[1]):int(panel[3]),
                                        int(panel[0]):int(panel[2])]

                f_out = open(
                    os.path.join(data_output_dir, file_name)+'.txt', 'w+')

                try:
                    cv2.imwrite(
                        os.path.join(data_output_dir, f'{file_name}.jpg'), 
                        panel_image)

                except:
                    print(f'{file_name} raise an error')
                    continue
                
                else:
                    lock.acquire()

                    total_list.append(
                        os.path.join('data/', f'{file_name}.jpg'))

                    lock.release()

                for d in digits:
                    class_id = d['class_id']
                    l,t,r,b = d['l'], d['t'], d['r'], d['b']

                    xmin = (l-panel[0]) / (panel[2]-panel[0])
                    ymin = (t-panel[1]) / (panel[3]-panel[1])
                    xmax = (r-panel[0]) / (panel[2]-panel[0])
                    ymax = (b-panel[1]) / (panel[3]-panel[1])
                        
                    c_x = (xmin+xmax)/2
                    c_y = (ymin+ymax)/2
                    w = xmax-xmin
                    h = ymax-ymin

                    f_out.write(f'{class_id} {c_x} {c_y} {w} {h}\n')
            
                f_out.close()


def generate_yolo_org_from_vott_multiprocess(vott_json, image_dir, output_dir,
                                             random_color=False,
                                             transform_num_per_panel=5, 
                                             valid_ratio=0.2,
                                             process_num=mp.cpu_count()):
                                
    with open(vott_json) as vott_buffer:
        vott = json.loads(vott_buffer.read())

    data_output_dir = os.path.join(output_dir, 'data/')
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    lock = mp.Manager().Lock()
    total_list = mp.Manager().list()

    partial_map = partial(_generate_worker, 
        image_dir=image_dir, 
        data_output_dir=data_output_dir, 
        random_color=random_color, 
        transform_num_per_panel=transform_num_per_panel,
        lock=lock,
        total_list=total_list)

    pool = mp.Pool(processes=process_num)
    result = pool.map(partial_map, list(vott['assets'].values())[::-1])

    total_list.sort()
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

            
if __name__ == '__main__':

    generate_yolo_org_from_vott_multiprocess(
        vott_json='/Users/rudy/Desktop/Development/Virtualenv/text-recognition/dataset/digits/renamed/target/vott-json-export/digits-export.json',
        image_dir='/Users/rudy/Desktop/Development/Virtualenv/text-recognition/dataset/digits/renamed/target/vott-json-export',
        output_dir='data_test/',
        transform_num_per_panel=100, 
        random_color=True,
        valid_ratio=0.2
    )


