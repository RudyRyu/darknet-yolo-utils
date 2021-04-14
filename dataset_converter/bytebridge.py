import os
import json
from collections import defaultdict
from pprint import pprint

import cv2


def show_rect(json_path, img_dir_path):
    with open(json_path) as json_buffer:
        result = json.loads(json_buffer.read())

    label_ddict = defaultdict(lambda: [])
    for k, v in result.items():
        read_label_list(v, label_ddict, k)

    for k, v in label_ddict.items():
        img_path = os.path.join(img_dir_path, k)
        img = cv2.imread(img_path)
        for label in v:
            # coords = json.loads(label['coordinates'])
            coords = label['coordinates']['coordinates']

            x1 = round(float(coords[0][0]))
            y1 = round(float(coords[0][1]))
            x2 = round(float(coords[2][0]))
            y2 = round(float(coords[2][1]))

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

        cv2.imshow(k, img)
        cv2.moveWindow(k, 20,20);
        cv2.waitKey()
        cv2.destroyAllWindows()


def read_label_list(list_value, label_ddict, key):
    for value in list_value:
        if value.__class__ == list:
            read_label_list(value, label_ddict, key)

        elif value.__class__ == dict:
            label_ddict[key].append(value)

show_rect(
    json_path='/Users/rav/Desktop/bytebridge/a3_front(compress)/express_a3_front_out.json',
    img_dir_path='/Users/rav/Desktop/bytebridge/a3_front(compress)/express_a3_front')
