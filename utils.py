import cv2


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
    return clone[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]


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
