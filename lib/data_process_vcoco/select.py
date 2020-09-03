import os
import h5py
import numpy as np
from tqdm import tqdm

import lib.utils.io as io
from lib.utils.bbox_utils import compute_area
from lib.dataset.coco_classes import COCO_CLASSES


def select_det_ids(boxes, scores, nms_keep_ids, score_thresh, max_dets):
    if nms_keep_ids is None:
        nms_keep_ids = np.arange(0, scores.shape[0])

    assert len(boxes) == len(scores) == len(nms_keep_ids)

    # Select non max suppressed dets

    sort_index = np.argsort(scores)[::-1]
    nms_scores = scores[sort_index]
    nms_boxes = boxes[sort_index]

    # Select dets above a score_thresh and which have area > 1
    nms_ids_above_thresh = np.nonzero(nms_scores > score_thresh)[0]
    nms_ids = []
    # print(nms_scores)

    for i in range(min(nms_ids_above_thresh.shape[0], max_dets)):
        area = compute_area(nms_boxes[i], invalid=-1)
        if area > 1:
            nms_ids.append(sort_index[i])

    print(len(nms_ids))  # index

    # If no dets satisfy previous criterion select the highest ranking one with area > 1
    if len(nms_ids) == 0:
        for i in range(nms_keep_ids.shape[0]):
            area = compute_area(nms_boxes[i], invalid=-1)
            if area > 1:
                nms_ids = [sort_index[i]]
                break

    # Convert nms ids to box ids
    nms_ids = np.array(nms_ids, dtype=np.int32)
    return nms_ids


def select_dets(
        boxes,
        scores,
        labels,
        nms_keep_indices,
        human_score_thresh, background_score_thresh, object_score_thresh,
        max_humans, max_background, max_objects_per_class):

    selected_dets = []

    start_end_ids = np.zeros([len(COCO_CLASSES), 2], dtype=np.int32)
    start_id = 0
    for cls_ind, cls_name in enumerate(COCO_CLASSES):

        cls_labels_index = np.where(labels == cls_ind)[0]
        if len(cls_labels_index) == 0:
            continue
        cls_boxes = boxes[cls_labels_index]
        cls_scores = scores[cls_labels_index]
        cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind-1])

        if cls_name == 'person':
            print("person")
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                human_score_thresh,
                max_humans)
        elif cls_name == 'background':
            assert False
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                object_score_thresh,
                max_objects_per_class)

        boxes_scores_rpn_id = np.concatenate((
            cls_boxes[select_ids],
            np.expand_dims(cls_scores[select_ids], 1),
            np.expand_dims(cls_nms_keep_ids[select_ids], 1)), 1)
        selected_dets.append(boxes_scores_rpn_id)
        num_boxes = boxes_scores_rpn_id.shape[0]
        start_end_ids[cls_ind, :] = [start_id, start_id + num_boxes]
        start_id += num_boxes

    selected_dets = np.concatenate(selected_dets)
    print(selected_dets.shape)

    return selected_dets, start_end_ids


def select_vcoco():
    select_boxes_dir = "data/vcoco"
    faster_rcnn_boxes = "data/vcoco/faster_rcnn_boxes"

    human_score_thresh = 0.01
    background_score_thresh = 0.01
    object_score_thresh = 0.01
    max_humans = 10
    max_background = 10
    max_objects_per_class = 10

    io.mkdir_if_not_exists(select_boxes_dir)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object("data/vcoco/annotations/anno_list_234.json")

    print('Creating selected_coco_cls_dets.hdf5 file ...')
    hdf5_file = os.path.join(select_boxes_dir, 'selected_coco_cls_dets.hdf5')
    f = h5py.File(hdf5_file, 'w')

    print('Selecting boxes ...')
    for anno in tqdm(anno_list):
        global_id = anno['global_id']
        # if not "test" in global_id:
        #     continue

        boxes_npy = os.path.join(
            faster_rcnn_boxes,
            f'{global_id}_boxes.npy')
        boxes = np.load(boxes_npy)

        scores_npy = os.path.join(
            faster_rcnn_boxes,
            f'{global_id}_scores.npy')
        scores = np.load(scores_npy)

        labels_npy = os.path.join(
            faster_rcnn_boxes,
            f'{global_id}_labels.npy')
        labels = np.load(labels_npy)

        assert len(labels) == len(scores) == len(boxes)

        nms_keep_indices_json = os.path.join(
            faster_rcnn_boxes,
            f'{global_id}_nms_keep_indices.json')
        nms_keep_indices = io.load_json_object(nms_keep_indices_json)
        assert len(nms_keep_indices) == 80

        selected_dets, start_end_ids = select_dets(boxes, scores, labels,
                                                   nms_keep_indices,
                                                   human_score_thresh, background_score_thresh, object_score_thresh,
                                                   max_humans, max_background, max_objects_per_class
                                                   )
        f.create_group(global_id)
        f[global_id].create_dataset('boxes_scores_rpn_ids', data=selected_dets)
        f[global_id].create_dataset('start_end_ids', data=start_end_ids)

    f.close()


if __name__ == "__main__":
    select_vcoco()

# python -m lib.data_process.select
