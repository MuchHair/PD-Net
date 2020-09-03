import os
import h5py
from tqdm import tqdm

import lib.utils.io as io

import numpy as np

import lib.utils.bbox_utils as bbox_utils


class BoxFeatures():
    def __init__(self):
        pass

    def compute_bbox_center(self, bbox):
        num_boxes = bbox.shape[0]
        center = np.zeros([num_boxes, 2])
        center[:, 0] = 0.5 * (bbox[:, 0] + bbox[:, 2])
        center[:, 1] = 0.5 * (bbox[:, 1] + bbox[:, 3])
        return center

    def normalize_center(self, c, im_wh):
        return c / im_wh

    def compute_l2_norm(self, v):
        return np.sqrt(np.sum(v ** 2))

    def compute_bbox_wh(self, bbox):
        num_boxes = bbox.shape[0]
        wh = np.zeros([num_boxes, 2])
        wh[:, 0] = 0.5 * (bbox[:, 2] - bbox[:, 0])
        wh[:, 1] = 0.5 * (bbox[:, 3] - bbox[:, 1])
        return wh

    def compute_offset(self, c1, c2, wh1, normalize):
        offset = c2 - c1
        if normalize:
            offset = offset / wh1
        return offset

    def compute_aspect_ratio(self, wh, take_log):
        aspect_ratio = wh[:, 0] / (wh[:, 1] + 1e-6)
        if take_log:
            aspect_ratio = np.log2(aspect_ratio + 1e-6)
        return aspect_ratio

    def compute_bbox_size_ratio(self, wh1, wh2, take_log):
        ratio = (wh2[:, 0] * wh2[:, 1]) / (wh1[:, 0] * wh1[:, 1])
        if take_log:
            ratio = np.log2(ratio + 1e-6)
        return ratio

    def compute_bbox_area(self, wh, im_wh, normalize):
        bbox_area = wh[:, 0] * wh[:, 1]
        if normalize:
            norm_factor = im_wh[:, 0] * im_wh[:, 1]
        else:
            norm_factor = 1
        bbox_area = bbox_area / norm_factor
        return bbox_area

    def compute_im_center(self, im_wh):
        return im_wh / 2

    def compute_features(self, bbox1, bbox2, im_wh):
        B = im_wh.shape[0]
        typical_wh = np.array([[640., 480.]])
        typical_wh = np.tile(typical_wh, (B, 1))
        im_c = self.compute_im_center(im_wh)
        c1 = self.compute_bbox_center(bbox1)
        c2 = self.compute_bbox_center(bbox2)
        c1_normalized = self.normalize_center(c1, im_wh)
        c2_normalized = self.normalize_center(c2, im_wh)
        wh1 = self.compute_bbox_wh(bbox1)
        wh2 = self.compute_bbox_wh(bbox2)
        aspect_ratio1 = self.compute_aspect_ratio(wh1, take_log=False)
        aspect_ratio2 = self.compute_aspect_ratio(wh2, take_log=False)
        area1 = self.compute_bbox_area(wh1, im_wh, normalize=True)
        area2 = self.compute_bbox_area(wh2, im_wh, normalize=True)
        area1_unnorm = self.compute_bbox_area(wh1, typical_wh, normalize=True)
        area2_unnorm = self.compute_bbox_area(wh2, typical_wh, normalize=True)
        area_im = self.compute_bbox_area(im_wh, typical_wh, normalize=True)
        offset_normalized = self.compute_offset(c1, c2, wh1, normalize=True)
        bbox_size_ratio = self.compute_bbox_size_ratio(wh1, wh2, take_log=False)
        iou = bbox_utils.compute_iou_batch(bbox1, bbox2)
        box_feat = np.concatenate((
            offset_normalized,
            c1_normalized - 0.5,
            c2_normalized - 0.5,
            iou[:, np.newaxis],
            aspect_ratio1[:, np.newaxis],  # w1/h1
            aspect_ratio2[:, np.newaxis],  # w2/h2
            bbox_size_ratio[:, np.newaxis],  # w2xh2 / w1xh1
            area1_unnorm[:, np.newaxis],  # w1xh1
            area2_unnorm[:, np.newaxis],  # w2xh2
            area1[:, np.newaxis],  # w1xh1 / imwximh
            area2[:, np.newaxis],  # w2xh2 / imwximh
            area_im[:, np.newaxis],  # imwximh
            wh1 / typical_wh,
            wh2 / typical_wh,
            im_wh / typical_wh), 1)
        return box_feat


def compute_box_feats(human_boxes, object_boxes, img_size):
    feat_extractor = BoxFeatures()
    num_cand = human_boxes.shape[0]
    imh, imw = [float(v) for v in img_size[:2]]
    im_wh = np.array([[imw, imh]], dtype=np.float32)
    im_wh = np.tile(im_wh, (num_cand, 1))
    feats = feat_extractor.compute_features(
        human_boxes,
        object_boxes,
        im_wh)
    return feats


def main(subset, exp_dir, hoi_cand_hdf5):
    hoi_cands = h5py.File(hoi_cand_hdf5, 'r')

    box_feats_hdf5 = os.path.join(
        exp_dir,
        f'hoi_candidates_box_feats_{subset}.hdf5')
    box_feats = h5py.File(box_feats_hdf5, 'w')

    anno_list = io.load_json_object("data/vcoco/annotations/anno_list.json")
    anno_dict = {anno['global_id']: anno for anno in anno_list}

    for global_id in tqdm(hoi_cands.keys()):
        img_hoi_cands = hoi_cands[global_id]
        human_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, :4]
        object_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, 4:8]
        img_size = anno_dict[global_id]['image_size'][:2]
        feats = compute_box_feats(human_boxes, object_boxes, img_size)
        box_feats.create_dataset(global_id, data=feats)

    box_feats.close()


# python -m lib.data_process.cache_box_features
if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        main("{subset}", "data/vcoco",
             f"data/vcoco/hoi_candidates_{subset}.hdf5")
