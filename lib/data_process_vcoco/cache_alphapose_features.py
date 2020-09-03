import os
import h5py

import numpy as np
from tqdm import tqdm

import lib.utils.io as io

import copy
import numpy as np


class PoseFeatures():
    def __init__(self, num_keypts=17):
        self.num_keypts = num_keypts

    def rpn_id_to_pose_h5py_to_npy(self, rpn_id_to_pose_h5py):
        rpn_id_to_pose_npy = {}
        for rpn_id in rpn_id_to_pose_h5py.keys():
            rpn_id_to_pose_npy[rpn_id] = rpn_id_to_pose_h5py[rpn_id][()]
        return rpn_id_to_pose_npy

    def rpn_id_to_pose_h5py_to_npy_cpn(self, rpn_id_to_pose_h5py):
        rpn_id_to_pose_npy = {}
        for rpn_id in rpn_id_to_pose_h5py.keys():
            rpn_id_to_pose_npy[rpn_id] = np.array(rpn_id_to_pose_h5py[rpn_id]["keypoints"]) \
                .reshape(self.num_keypts, 3)
        return rpn_id_to_pose_npy

    def get_keypoints(self, rpn_ids, rpn_id_to_pose):
        num_cand = rpn_ids.shape[0]
        keypts = np.zeros([num_cand, 17, 3])
        for i in range(num_cand):
            rpn_id = str(int(rpn_ids[i]))
            keypts_ = rpn_id_to_pose[rpn_id]
            keypts[i] = keypts_
        return keypts

    def compute_bbox_wh(self, bbox):
        num_boxes = bbox.shape[0]
        wh = np.zeros([num_boxes, 2])
        wh[:, 0] = (bbox[:, 2] - bbox[:, 0])
        wh[:, 1] = (bbox[:, 3] - bbox[:, 1])
        return wh

    def encode_pose(self, keypts, human_box):
        wh = self.compute_bbox_wh(human_box)  # Bx2
        wh = np.tile(wh[:, np.newaxis, :], (1, self.num_keypts, 1))  # Bx18x2
        xy = np.tile(human_box[:, np.newaxis, :2], (1, self.num_keypts, 1))  # Bx18x2
        pose = copy.deepcopy(keypts)  # Bx18x3
        pose[:, :, :2] = (pose[:, :, :2] - xy) / (wh + 1e-6)
        return pose

    def encode_relative_pose(self, keypts, object_box, im_wh):
        keypts_ = copy.deepcopy(keypts)
        keypts_[:, :, :2] = keypts_[:, :, :2] / im_wh
        x1y1 = object_box[:, :2]
        x1y1 = np.tile(x1y1[:, np.newaxis, :], (1, self.num_keypts, 1))
        x1y1 = x1y1 / im_wh
        x2y2 = object_box[:, 2:4]
        x2y2 = np.tile(x2y2[:, np.newaxis, :], (1, self.num_keypts, 1))
        x2y2 = x2y2 / im_wh
        x1y1_wrt_keypts = x1y1 - keypts_[:, :, :2]  # Bx18x2
        x2y2_wrt_keypts = x2y2 - keypts_[:, :, :2]  # Bx18x2
        return x1y1_wrt_keypts, x2y2_wrt_keypts

    def compute_pose_feats(
            self,
            human_bbox,
            object_bbox,
            rpn_ids,
            rpn_id_to_pose,
            im_wh):
        B = human_bbox.shape[0]
        im_wh = np.tile(im_wh[:, np.newaxis, :], (1, self.num_keypts, 1))
        keypts = self.get_keypoints(rpn_ids, rpn_id_to_pose)
        absolute_pose = self.encode_pose(keypts, human_bbox)  # Bx18x3
        keypts_conf = absolute_pose[:, :, 2][:, :, np.newaxis]  # Bx18x1
        absolute_pose = np.reshape(absolute_pose, (B, -1))  # Bx54
        x1y1_wrt_keypts, x2y2_wrt_keypts = self.encode_relative_pose(
            keypts,
            object_bbox,
            im_wh)
        relative_pose = np.concatenate((
            x1y1_wrt_keypts,
            x2y2_wrt_keypts,
            keypts_conf), 2)
        relative_pose = np.reshape(relative_pose, (B, -1))
        feats = {
            'absolute_pose': absolute_pose,  # 51
            'relative_pose': relative_pose,  # Bx90 (17*2 + 17*2 + 17)
        }
        return feats


def main(subset, exp_dir, hoi_cand_hdf5, pose_dict_file):
    hoi_cands = h5py.File(hoi_cand_hdf5, 'r')
    pose_dict = io.load_json_object(pose_dict_file)

    human_pose_feats_hdf5 = os.path.join(
        exp_dir,
        f'human_pose_feats_{subset}_alpha_bbox.hdf5')
    human_pose_feats = h5py.File(human_pose_feats_hdf5, 'w')

    anno_list = io.load_json_object("data/vcoco/annotations/anno_list.json")
    anno_dict = {anno['global_id']: anno for anno in anno_list}

    pose_feat_computer = PoseFeatures(num_keypts=17)
    for global_id in tqdm(hoi_cands.keys()):
        img_hoi_cands = hoi_cands[global_id]
        human_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, :4]
        object_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, 4:8]
        human_rpn_ids = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, 10]

        key = "COCO_val2014_" + global_id.split("_")[1].zfill(12) + ".jpg"

        rpn_id_to_pose = pose_feat_computer.rpn_id_to_pose_h5py_to_npy_cpn(
            pose_dict[key])

        img_size = anno_dict[global_id]['image_size'][:2]
        imh, imw = [float(v) for v in img_size[:2]]
        im_wh = np.array([[imw, imh]], dtype=np.float32)
        num_cand = human_boxes.shape[0]
        im_wh = np.tile(im_wh, (num_cand, 1))
        feats = pose_feat_computer.compute_pose_feats(
            human_boxes,
            object_boxes,
            human_rpn_ids,
            rpn_id_to_pose,
            im_wh)
        human_pose_feats.create_group(global_id)
        human_pose_feats[global_id].create_dataset(
            'absolute_pose',
            data=feats['absolute_pose'])
        human_pose_feats[global_id].create_dataset(
            'relative_pose',
            data=feats['relative_pose'])

    human_pose_feats.close()


# python -m lib.data_process.cache_alphapose_features
if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        main(f"{subset}", f"data/vcoco",
             f"data/vcoco/hoi_candidates_{subset}.hdf5",
             f"data/vcoco/alphapose-results_{subset}.json")
