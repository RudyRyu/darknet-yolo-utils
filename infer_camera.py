import math
import time

import cv2
import numpy as np

import image_selection

def cbox2bbox(cbox):
    w_half = cbox[2] / 2
    h_half = cbox[3] / 2
    x1 = cbox[0] - w_half
    y1 = cbox[1] - h_half
    x2 = cbox[0] + w_half
    y2 = cbox[1] + h_half
    return [x1, y1, x2, y2]

def bbox2cbox(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] - w / 2
    cy = bbox[1] - h / 2
    return [cx, cy, w, h]

def calc_iou(bb1, bb2):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(bb1[0], bb2[0])
	yA = max(bb1[1], bb2[1])
	xB = min(bb1[2], bb2[2])
	yB = min(bb1[3], bb2[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
	boxBArea = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


# def detect(net, output_names, img):
#     blob = cv2.dnn.blobFromImage(img, 1./255.)
#     net.setInput(blob)

#     detections = net.forward(output_names)
#     bboxes = []
#     scores = []
#     img_h = img.shape[0]
#     img_w = img.shape[1]

#     print(detections)
#     for output in detections:
#         # loop over each of the detections

#         bbox = output[..., 0:4]
#         obj = output[..., 4]
#         cls = output[..., 5]

#         for idx in np.argwhere(obj >= 0.5):
#             b = bbox[idx][0] * [img_w, img_h, img_w, img_h]
#             x = int((b[0] - b[2] // 2))
#             y = int((b[1] - b[3] // 2))
#             w = int(b[2])
#             h = int(b[3])
#             bboxes.append([x, y, w, h])
#             scores.append(float(cls[idx]))
#     indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.5, 0.5)
#     bboxes_final = []
#     for idx in indices:
#         (x, y, w, h) = bboxes[idx[0]]
#         cx = x + w / 2
#         cy = y + h / 2
#         bboxes_final.append([cx, cy, w, h])

#     return bboxes_final

def detect(net, output_names, img, score_thresh=0.5):
    blob = cv2.dnn.blobFromImage(img, 1./255.)
    net.setInput(blob)

    detections = net.forward(output_names)
    H = img.shape[0]
    W = img.shape[1]

    boxes = []
    confidences = []
    classIDs = []

    for output in detections:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > score_thresh:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)

    return idxs, boxes, confidences, classIDs

def draw_rect(img, bb, size, color):
    x = bb[0] - bb[2] // 2 - 5
    y = bb[1] - bb[3] // 2 - 5
    w = bb[2] + 10
    h = bb[3] + 10
    x = x if x >= 0 else 0
    y = y if y >= 0 else 0
    w = w if x+w < size[0] else size[0]-x
    h = h if y+h < size[1] else size[1]-y
    cv2.rectangle(img, (x,y), (x+w,y+h), color,4)

    return img


def crop_image(image, roi):
    #print roi
    clone = image.copy()
    retImg = clone[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    return retImg


def combine_result(idxs, boxes, class_ids, labels):

    x_and_class_list = []

    if len(idxs) <= 0:
        return ''

    for i in idxs.flatten():
        x_and_class_list.append((boxes[i][0], labels[class_ids[i]]))

    x_and_class_str_list = sorted(x_and_class_list, key=lambda tup: tup[0])

    result = ''
    for _, class_str in x_and_class_str_list:
        result += class_str

    return result

def detect_rois(image, roi_points, roi_size_wh, net, output_names, 
                score_thresh):
    
    idxs_list, boxes_list, scores_list, class_ids_list = [], [], [] ,[]
    for roi in range(int(len(roi_points)/2)):
        sel = crop_image(image.copy(),
                         [roi_points[roi*2],
                          roi_points[2*roi+1]])

        sel = cv2.resize(sel, (roi_size_wh[0], roi_size_wh[1]))
        idxs, boxes, scores, class_ids = detect(net, output_names, 
                                                sel, score_thresh)

        idxs_list.append(idxs)
        boxes_list.append(boxes)
        scores_list.append(scores)
        class_ids_list.append(class_ids)

    return idxs_list, boxes_list, scores_list, class_ids_list


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

    roi_points = image_selection.getSelectionsFromImage(frame)

    frame_num = 0
    while True:
        _, frame = cap.read()
        frame_num += 1
        if frame_num % frame_interval != 0:
            continue
        
        start_time = time.time()

        frame = cv2.resize(frame, video_size_wh)
        idxs_list, boxes_list, scores_list, class_ids_list = \
            detect_rois(
                image=frame, 
                roi_points=roi_points, roi_size_wh=roi_size_wh,
                net=net, output_names=output_names, score_thresh=score_thresh)
        
        for r, idxs, boxes, scores, class_ids in zip(
                                                range(int(len(roi_points)/2)), 
                                                idxs_list, boxes_list, 
                                                scores_list, class_ids_list):

            sel = crop_image(frame.copy(), [roi_points[r*2], 
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

            combined = combine_result(idxs, boxes, class_ids, labels=labels)
            cv2.rectangle(frame, roi_points[r*2], roi_points[r*2+1],
                          (0,255,0), 2)

            cv2.putText(frame, combined, roi_points[r*2],
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

            # show the output image
        fps = 1/(time.time()-start_time)
        fps = str(int(fps)) + ' fps'

        cv2.putText(frame, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey()
            

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
            boxes = detect(net, output_names, img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fps = 1/(time.time()-start_time)
            fps = str(int(fps)) + ' fps'
            for b in boxes:
                b = np.array(b, dtype=np.int32)
                img = draw_rect(img, b, video_size_wh, (0, 0, 255))

            cv2.putText(img, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 2, cv2.LINE_AA)

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


def detect_image(cfg, weights, image_path, image_size_wh, label_path, score_thresh, output_image_path):
    
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
    idxs, boxes, confidences, classIDs = detect(net, output_names, image, score_thresh)
    fps = 1/(time.time()-start_time)
    fps = str(int(fps)) + ' fps'

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = '{}'.format(LABELS[classIDs[i]])
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
        cfg='cfg/digit/sh_digit_x2.cfg',
        weights='cfg/digit/sh_digit_x2_best.weights',
        label_path='cfg/digit/digit.names',
        video_path='/Users/rudy/Desktop/output-cut.mp4', 
        video_size_wh=(1920/2, 1080/2), 
        roi_size_wh=(128,64), 
        score_thresh=0.5
    )

