import multiprocessing as mp
import os
import time
from functools import partial
from itertools import repeat

import cv2
import numpy as np

import utils
import infer


def _evaluate_worker(image_path, cfg, weights, output_names, input_size_wh, 
                     score_thresh, iou_thresh, labels, save_size_wh, 
                     save_dir_path):

    """
    Need to add False Positive
    """

    image_path = image_path.strip()
    image = cv2.imread(image_path)
    net = cv2.dnn.readNetFromDarknet(cfg, weights)

    idxs, boxes, scores, class_ids = \
        infer.infer_image(image, net, output_names, input_size_wh, score_thresh)

    label_filename = os.path.splitext(image_path)[0] + '.txt'

    all_test_pass = True
    f_label = list(open(label_filename, 'r'))
    for label in f_label:
        label = label.strip().split(' ')
        t_class_id = int(label[0])

        t_cbox = np.array(tuple(map(float, label[1:5])))
        t_cbox = t_cbox * np.array([*input_size_wh, *input_size_wh])
        t_bbox = utils.cbox2bbox(t_cbox)

        label_test_pass = False

        if len(idxs) >= 1:
            for i in idxs.flatten():
                if t_class_id != class_ids[i]:
                    continue
                
                cbox = boxes[i][:4] * np.array([*input_size_wh, *input_size_wh])
                bbox = utils.cbox2bbox(cbox)
                iou = utils.calc_iou(bbox, t_bbox)
                
                if iou >= iou_thresh:
                    label_test_pass = True
                    break

        if not label_test_pass:
            all_test_pass = False
            break

    if not all_test_pass:
        save_image = cv2.resize(image, save_size_wh)
        save_image_true = save_image.copy()
        save_image_pred = save_image.copy()

        for label in f_label:
            label = label.strip().split(' ')
            t_class_id = int(label[0])

            t_cbox = np.array(tuple(map(float, label[1:5])))
            t_cbox = t_cbox * np.array([*save_size_wh, *save_size_wh])
            t_x1, t_y1, t_x2, t_y2 = list(map(int, utils.cbox2bbox(t_cbox)))

            cv2.rectangle(
                save_image_true, (t_x1,t_y1), (t_x2,t_y2), (0,255,0), 2)
                
            text = f'{labels[t_class_id]}'
            cv2.putText(
                save_image_true, text, (t_x1+5, t_y2-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        
        if len(idxs) >= 1:
            for i in idxs.flatten():
                cbox = boxes[i][:4] * np.array([*save_size_wh, *save_size_wh])

                x1, y1, x2, y2 = list(map(int, utils.cbox2bbox(cbox)))

                cv2.rectangle(save_image_pred, (x1,y1), (x2,y2), (0,255,0), 2)

                label = f'{labels[class_ids[i]]}'
                cv2.putText(
                    save_image_pred, label, (x1+5, y2-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                score = f'{scores[i]:.2f}'
                cv2.putText(
                    save_image_pred, score, (x1+5, y2+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        stack_image = np.vstack((save_image, save_image_true, save_image_pred))

        cv2.imwrite(
            os.path.join(save_dir_path, os.path.basename(image_path)), 
            stack_image)

        return 1
    
    return 0


def evaluate(cfg, weights, label_path, image_paths_txt, input_size_wh, 
             score_thresh, iou_thresh, save_size_wh, save_dir_path,
             process_num=mp.cpu_count()):

    """
    1. image / label_dict batch list 구성
    2. multiprocessing
        - infer image / images 속도 비교해보고 빠른 걸로
            => inference가 multiprocessing으로 동작하는지 먼저 확인해야함
            => 동작한다면 단일 이미지 연산 먼저해보기
        - label을 기준으로 iou 비교
        - iou 0.5 이하 일 경우 에러로 판단
        - infer 결과 이미지 저장

    """
    labels = open(label_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    
    output_names = []
    for name in net.getLayerNames():
        if 'yolo' in name:
            output_names.append(name)

    image_paths = list(open(image_paths_txt, 'r'))
    
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    _partial_map = partial(_evaluate_worker, 
        cfg=cfg, weights=weights, labels=labels, output_names=output_names, 
        input_size_wh=input_size_wh, 
        score_thresh=score_thresh, iou_thresh=iou_thresh, 
        save_size_wh=save_size_wh, save_dir_path=save_dir_path)

    pool = mp.Pool(processes=process_num)
    result = pool.map(_partial_map, image_paths)

    total_num = len(result)
    valid_num = len(result) - sum(result)

    print(f'total image: {total_num}')
    print(f'valid image: {valid_num}')
    print(f'precision: {valid_num / total_num:0.3f}')


if __name__ == '__main__':
    evaluate(
        cfg='cfg/digit/sh_digit.cfg',
        weights='cfg/digit/sh_digit_best.weights',
        label_path='cfg/digit/digit.names',
        image_paths_txt='digit_data/valid.txt',
        input_size_wh=(128,64),
        score_thresh=0.7, 
        iou_thresh=0.3,
        save_size_wh=(256, 128),
        save_dir_path='result/'
        process_num=mp.cpu_count()
    )
