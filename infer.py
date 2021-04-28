import math
import time
from collections import defaultdict

import cv2
import numpy as np

import image_selection
import utils


def infer_image(net, image, output_names, input_size_wh=None, score_thresh=0.5):
    
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
        print('cbox', cbox)
        x = int(cbox[0] - (cbox[2] / 2))
        y = int(cbox[1] - (cbox[3] / 2))
        xywh_boxes.append([x,y, cbox[2], cbox[3]])

    idxs = cv2.dnn.NMSBoxes(xywh_boxes, scores, score_thresh, 0.5)

    return idxs, boxes, scores, class_ids


def infer_images(net, images, input_size_wh, output_names, score_thresh=0.5):

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
                xywh_boxes, scores_dict[b], score_thresh, 0.5))

    return batch_idxs, batch_boxes, batch_scores, batch_class_ids


def detect_rois(image, roi_points, roi_size_wh, net, output_names, 
                score_thresh):
    
    idxs_list, boxes_list, scores_list, class_ids_list = [], [], [] ,[]
    for roi in range(int(len(roi_points)/2)):
        sel = utils.crop_image(image.copy(), [roi_points[roi*2],
                                              roi_points[2*roi+1]])

        idxs, boxes, scores, class_ids = infer_image(net, sel, output_names, 
                                                     input_size_wh=roi_size_wh,
                                                     score_thresh=score_thresh)

        idxs_list.append(idxs)
        boxes_list.append(boxes)
        scores_list.append(scores)
        class_ids_list.append(class_ids)

    return idxs_list, boxes_list, scores_list, class_ids_list


def detect_rois_batch_inference(image, roi_points, roi_size_wh, net, 
                                output_names, score_thresh):
    
    sels = []
    for roi in range(int(len(roi_points)/2)):
        sel = utils.crop_image(image.copy(), [roi_points[roi*2],
                                              roi_points[2*roi+1]])
        sels.append(sel)

    batch_idxs, batch_boxes, batch_scores, batch_class_ids = \
        infer_images(net, sels, roi_size_wh, output_names, score_thresh)

    return batch_idxs, batch_boxes, batch_scores, batch_class_ids


def detect_video_with_roi(cfg, weights, video_path, video_size_wh, roi_size_wh, 
                          label_path, score_thresh, frame_interval=30):
    
    labels = open(label_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    output_names = []
    for name in net.getLayerNames():
        if 'yolo' in name:
            output_names.append(name)

    print(output_names)

    cap = cv2.VideoCapture(video_path) # Open video
    #cap = cv2.VideoCapture("rtsp://admin:1234@ijoon.net:30084/h264") # Open video
    # cap.set(3, video_size_wh[0])
    # cap.set(4, video_size_wh[1])

    _, frame = cap.read() # Get first frame
    frame = cv2.resize(frame, video_size_wh)

    # roi_points = image_selection.getSelectionsFromImage(frame)
    roi_points = [(19, 516), (103, 561), (36, 585), (121, 626), 
                  (49, 652), (132, 690), (145, 507), (237, 554), 
                  (158, 580), (246, 623), (175, 647), (262, 685), 
                  (342, 496), (444, 539), (353, 571), (452, 612), 
                  (368, 640), (463, 682), (495, 487), (600, 532), 
                  (504, 563), (608, 605), (511, 637), (616, 679), 
                  (730, 475), (842, 517), (735, 552), (844, 594), 
                  (741, 626), (849, 667), (899, 465), (1009, 510), 
                  (905, 543), (1012, 585), (907, 619), (1014, 658), 
                  (1147, 456), (1252, 500), (1144, 532), (1250, 572), 
                  (1143, 605), (1248, 648), (1307, 448), (1410, 492), 
                  (1302, 524), (1406, 567), (1300, 594), (1401, 636), 
                  (1774, 544), (1875, 586), (1741, 708), (1805, 735), 
                  (132, 851), (211, 905), (139, 891), (218, 938)]

    # print(roi_points)
    frame_num = -1
    while True:
        _, frame = cap.read()
        frame_num += 1
        if frame_num % frame_interval != 0:
            continue
        
        start_time = time.time()

        frame = cv2.resize(frame, video_size_wh)
        idxs_list, boxes_list, scores_list, class_ids_list = \
            detect_rois_batch_inference(
                image=frame, roi_points=roi_points, roi_size_wh=roi_size_wh,
                net=net, output_names=output_names, score_thresh=score_thresh)
        
        fps = 1/(time.time()-start_time)
        fps = f'{fps:.2f} fps'

        for r, idxs, boxes, scores, class_ids in zip(
                                                range(int(len(roi_points)/2)), 
                                                idxs_list, boxes_list, 
                                                scores_list, class_ids_list):

            sel = utils.crop_image(frame.copy(), [roi_points[r*2], 
                                                  roi_points[r*2+1]])

            sel = cv2.resize(sel, (roi_size_wh[0], roi_size_wh[1]))

            sel_show_size_wh = (192, 96)

            # loop over the indexes we are keeping

            if len(idxs) <= 0:
                cv2.imshow(f'panel_{r}', cv2.resize(sel, sel_show_size_wh))
            
            else:
                for i in idxs.flatten():
                    sel = cv2.resize(sel, sel_show_size_wh)
                    
                    w_ratio = sel_show_size_wh[0] / roi_size_wh[0]
                    h_ratio = sel_show_size_wh[1] / roi_size_wh[1]
                    
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    x, y = int(x*w_ratio), int(y*h_ratio)
                    w, h = int(w*w_ratio), int(h*h_ratio)

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in colors[class_ids[i]]]
                    cv2.rectangle(sel, (x, y), (x + w, y + h), color, 2)
                    text = "{}:{:.2f}".format(labels[class_ids[i]], scores[i])
                    # text = '{}'.format(LABELS[class_ids[i]])
                    cv2.putText(
                        sel, text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1)
                    cv2.imshow(f'panel_{r}', sel)

            combined = utils.combine_result(idxs, boxes, class_ids, labels)

            cv2.rectangle(frame, roi_points[r*2], roi_points[r*2+1],
                          (0,255,0), 2)

            cv2.putText(frame, combined, roi_points[r*2],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

            # show the output image
        
        cv2.putText(frame, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX,  
            1, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if frame_num == 0:
            cv2.waitKey()

        else:
            cv2.waitKey(1)
            # cv2.waitKey()
            

def detect_video(cfg, weights, video_path, video_size_wh, output_video_path):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    output_names = []
    for name in net.getLayerNames():
        if 'yolo' in name:
            output_names.append(name)
    print(output_names)
    output_names = output_names[1:]
    cap = cv2.VideoCapture(video_path)

    if output_video_path:
        writer = cv2.VideoWriter(filename=output_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=30,
            frameSize=video_size_wh)

    count = 1
    fps_count = 0
    while cap.isOpened():
        for _ in range(count):
            is_cap, img = cap.read()

        fps_count += 1
        count = 1

        if is_cap:
            if fps_count % 10 != 0:
                continue

            start_time = time.time()
            img = cv2.resize(img, video_size_wh)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = infer_image(net, output_names, img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fps = 1/(time.time()-start_time)
            fps = str(int(fps)) + ' fps'
            for b in boxes:
                b = np.array(b, dtype=np.int32)
                img = utils.draw_rect(img, b, video_size_wh, (0, 0, 255))

            cv2.putText(img, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (100, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('results', img)

            if output_video_path:
                writer.write(img)

            ch = cv2.waitKey(1) & 0xFF
            if ch == 27: # Exit with ESC Key
                break
            elif ch == 32:
                count = 120
        else:
            break

    cap.release()
    if output_video_path:
        writer.release()


def detect_image(cfg, weights, image_path, image_size_wh, label_path, 
                 score_thresh, output_image_path):
    
    LABELS = open(label_path).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    output_names = []
    for name in net.getLayerNames():
        if 'yolo' in name:
            output_names.append(name)

    print(output_names)

    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size_wh)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', image)

    start_time = time.time()
    idxs, boxes, scores, class_ids = infer_image(net, output_names, image, 
                                                 score_thresh)
    fps = 1/(time.time()-start_time)
    fps = str(int(fps)) + ' fps'

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            # text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            text = '{}'.format(LABELS[class_ids[i]])
            cv2.putText(image, text, (x+1, y+h-3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)

        # show the output image
        cv2.imshow("Image", cv2.resize(image, (192, 96)))
        cv2.waitKey()

    # for b in boxes:
    #     b = np.array(b, dtype=np.int32)
    #     img = draw_rect(img, b, image_size_wh, (0, 0, 255))

    # cv2.putText(img, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('results', img)
    # cv2.waitKey()


if __name__ == '__main__':

    # detect_image(
    #     cfg='cfg/digit/sh_digit_x2.cfg',
    #     weights='cfg/digit/sh_digit_x2_best.weights',
    #     image_path='sample/test_images/test_image12.png',
    #     image_size_wh=(128,64),
    #     label_path='cfg/digit/digit.names',
    #     score_thresh=0.5,
    #     output_image_path=''
    # )

    detect_video_with_roi(
        cfg='cfg/digit/sh_digit.cfg',
        weights='cfg/digit/sh_digit_best.weights',
        label_path='cfg/digit/digit.names',
        video_path='/Users/rudy/Desktop/output-cut.mp4',
        video_size_wh=(1920, 1080),
        roi_size_wh=(128,64),
        score_thresh=0.5
    )

