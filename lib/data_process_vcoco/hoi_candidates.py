import os
import numpy as \
    np
from tqdm import tqdm
import threading
import h5py

import lib.utils.io as io
from lib.dataset.coco_classes import COCO_CLASSES


class HoiCandidatesGenerator():
    def __init__(self):
        self.hoi_classes = self.get_hoi_classes()

    def get_hoi_classes(self):
        hoi_list = io.load_json_object("data/vcoco/annotations/hoi_list_234.json")
        print(len(hoi_list))
        hoi_classes = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_classes

    def predict(self, selected_dets):

        pred_hoi_dets = []
        start_end_ids = np.zeros([len(self.hoi_classes), 2], dtype=np.int32)
        start_id = 0
        # hoi_id begin at 1
        for hoi_id, hoi_info in self.hoi_classes.items():
            dets = self.predict_hoi(selected_dets, hoi_info)

            pred_hoi_dets.append(dets)
            hoi_idx = int(hoi_id) - 1

            start_end_ids[hoi_idx, :] = [start_id, start_id + dets.shape[0]]
            start_id += dets.shape[0]

        pred_hoi_dets = np.concatenate(pred_hoi_dets)
        return pred_hoi_dets, start_end_ids

    def predict_hoi(self, selected_dets, hoi_info):
        hoi_object = hoi_info['object']

        human_boxes = selected_dets['boxes']['person']
        human_scores = selected_dets['scores']['person']
        human_rpn_ids = selected_dets['rpn_ids']['person']

        object_boxes = selected_dets['boxes'][hoi_object]
        object_scores = selected_dets['scores'][hoi_object]
        object_rpn_ids = selected_dets['rpn_ids'][hoi_object]

        num_hoi_dets = human_boxes.shape[0] * object_boxes.shape[0]

        hoi_dets = np.zeros([num_hoi_dets, 13])
        hoi_idx = int(hoi_info['id']) - 1
        hoi_dets[:, -1] = hoi_idx
        count = 0
        for i in range(human_boxes.shape[0]):
            for j in range(object_boxes.shape[0]):
                hoi_dets[count, :4] = human_boxes[i]
                hoi_dets[count, 4:8] = object_boxes[j]
                hoi_dets[count, 8:12] = [human_scores[i], object_scores[j], \
                                         human_rpn_ids[i], object_rpn_ids[j]]
                count += 1

        return hoi_dets


def generate(exp_dir, selected_dets_hdf5, subset):
    print(f'Reading split_ids.json ...')
    split_ids = io.load_json_object("data/vcoco/annotations/split_ids.json.json")

    print('Creating an object-detector-only HOI detector ...')
    hoi_cand_gen = HoiCandidatesGenerator()

    print(f'Creating a hoi_candidates_{subset}.hdf5 file ...')
    hoi_cand_hdf5 = os.path.join(
        exp_dir, f'hoi_candidates_{subset}.hdf5')
    f = h5py.File(hoi_cand_hdf5, 'w')

    print('Reading selected dets from hdf5 file ...')
    all_selected_dets = h5py.File(selected_dets_hdf5, 'r')

    for global_id in tqdm(split_ids[subset]):
        global_id = subset + "_" + global_id
        selected_dets = {
            'boxes': {},
            'scores': {},
            'rpn_ids': {}
        }
        start_end_ids = all_selected_dets[global_id]['start_end_ids'].value
        boxes_scores_rpn_ids = \
            all_selected_dets[global_id]['boxes_scores_rpn_ids'].value

        for cls_ind, cls_name in enumerate(COCO_CLASSES):
            start_id, end_id = start_end_ids[cls_ind]
            boxes = boxes_scores_rpn_ids[start_id:end_id, :4]
            scores = boxes_scores_rpn_ids[start_id:end_id, 4]
            rpn_ids = boxes_scores_rpn_ids[start_id:end_id, 5]
            selected_dets['boxes'][cls_name] = boxes
            selected_dets['scores'][cls_name] = scores
            selected_dets['rpn_ids'][cls_name] = rpn_ids

        pred_dets, start_end_ids = hoi_cand_gen.predict(selected_dets)
        f.create_group(global_id)
        f[global_id].create_dataset(
            'boxes_scores_rpn_ids_hoi_idx', data=pred_dets)
        f[global_id].create_dataset('start_end_ids', data=start_end_ids)

    f.close()


# python -m lib.data_process.hoi_candidates
if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        generate("data/vcoco", "data/vcoco/selected_coco_cls_dets.hdf5", f"{subset}")
