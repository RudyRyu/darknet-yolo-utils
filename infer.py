import time
from collections import defaultdict

import cv2
import numpy as np

import utils


def infer_image(image, net, output_names, input_size_wh=None, score_thresh=0.5):
    
    if input_size_wh:
        blob = cv2.dnn.blobFromImage(image, 1./255., input_size_wh)
    else:
        blob = cv2.dnn.blobFromImage(image, 1./255.)

    net.setInput(blob)

    detections = net.forward(output_names)
    boxes = []
    scores = []
    class_ids = []

    for detection in detections:
        # loop over each of the detections

        for output in detection:
            confidences = output[5:]
            class_id = np.argmax(confidences)
            score = confidences[class_id]

            if score <= score_thresh:
                continue

            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            # box = output[0:4] * np.array([*input_size_wh, *input_size_wh])
            # (centerX, centerY, width, height) = box.astype("int")
            # # use the center (x, y)-coordinates to derive the top and
            # # and left corner of the bounding box
            # x = int(centerX - (width / 2))
            # y = int(centerY - (height / 2))
            # # update our list of bounding box coordinates, confidences,
            # # and class IDs
            # boxes.append([x, y, int(width), int(height)])
            boxes.append(output[:4])
            scores.append(float(score))
            class_ids.append(class_id)

    xywh_boxes = []
    for cbox in boxes:
        cbox = cbox * np.array([*input_size_wh, *input_size_wh])
        x = int(cbox[0] - (cbox[2] / 2))
        y = int(cbox[1] - (cbox[3] / 2))
        xywh_boxes.append([x,y, cbox[2], cbox[3]])

    idxs = cv2.dnn.NMSBoxes(xywh_boxes, scores, score_thresh, 0.3)

    return idxs, boxes, scores, class_ids


def infer_images(images, net, input_size_wh, output_names, score_thresh=0.5):

    blob = cv2.dnn.blobFromImages(images, 1./255., input_size_wh)
    net.setInput(blob)

    detections = net.forward(output_names)
    
    boxes_dict = defaultdict(lambda: [])
    scores_dict = defaultdict(lambda: [])
    class_ids_dict = defaultdict(lambda: [])
    for batch_detection in detections:
        for b, detection in enumerate(batch_detection):
            for output in detection:
                confidences = output[5:]
                class_id = np.argmax(confidences)
                score = confidences[class_id]

                if score <= score_thresh:
                    continue

                # # scale the bounding box coordinates back relative to the
                # # size of the image, keeping in mind that YOLO actually
                # # returns the center (x, y)-coordinates of the bounding
                # # box followed by the boxes' width and height
                # box = output[0:4] * np.array([*input_size_wh, *input_size_wh])
                # (centerX, centerY, width, height) = box.astype("int")
                # # use the center (x, y)-coordinates to derive the top and
                # # and left corner of the bounding box
                # x = int(centerX - (width / 2))
                # y = int(centerY - (height / 2))
                # # update our list of bounding box coordinates, confidences,
                # # and class IDs
                # boxes_dict[b].append([x, y, int(width), int(height)])
                boxes_dict[b].append(output[:4])
                scores_dict[b].append(float(score))
                class_ids_dict[b].append(class_id)


    batch_boxes = []
    batch_scores = []
    batch_class_ids = []
    batch_idxs = []
    for b in range(len(detections[0])):
        batch_boxes.append(boxes_dict[b])
        batch_scores.append(scores_dict[b])
        batch_class_ids.append(class_ids_dict[b])

        xywh_boxes = []
        for cbox in boxes_dict[b]:
            cbox = cbox * np.array([*input_size_wh, *input_size_wh])
            x = int(cbox[0] - (cbox[2] / 2))
            y = int(cbox[1] - (cbox[3] / 2))
            xywh_boxes.append([x,y, cbox[2], cbox[3]])

        batch_idxs.append(
            cv2.dnn.NMSBoxes(
                xywh_boxes, scores_dict[b], score_thresh, 0.3))

    return batch_idxs, batch_boxes, batch_scores, batch_class_ids
