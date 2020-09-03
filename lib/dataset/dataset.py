import h5py
import numpy as np
from torch.utils.data import Dataset
import utils.io as io


class Features_PD_VCOCO(Dataset):
    def __init__(self, subset="trainval", fp_to_tp_ratio=1000, dir="vcoco"):
        self.subset = subset
        self.fp_to_tp_ratio = fp_to_tp_ratio
        assert subset == "trainval" or subset == "test"

        if self.subset == "trainval":
            self.hoi_cands_train = self.load_hdf5_file(f"data/{dir}/hoi_candidates_train.hdf5")
            self.hoi_cands_val = self.load_hdf5_file(f"data/{dir}/hoi_candidates_val.hdf5")

            self.hoi_cand_labels_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_train.hdf5")
            self.hoi_cand_labels_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_val.hdf5")
            self.hoi_cand_nis_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_train.hdf5")
            self.hoi_cand_nis_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_val.hdf5")

            self.box_feats_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_train.hdf5")
            self.box_feats_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_val.hdf5")

            self.human_pose_feat_train = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_train_bbox.hdf5")
            self.human_pose_feat_val = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_val_bbox.hdf5")

            self.hoi_cands_union_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_train.hdf5")
            self.hoi_cands_union_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_val.hdf5")

            self.human_new_features_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_train.hdf5")
            self.human_new_features_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_val.hdf5")
            self.object_new_features_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_train.hdf5")
            self.object_new_features_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_val.hdf5")

            self.global_ids_train = self.load_subset_ids("train")
            self.global_ids_val = self.load_subset_ids("val")
            self.global_ids = self.global_ids_train + self.global_ids_val
            print(len(self.global_ids))

        else:
            self.hoi_cands_test = self.load_hdf5_file(f"data/{dir}/hoi_candidates_test.hdf5")

            self.hoi_cand_labels_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_test.hdf5")
            self.hoi_cand_nis_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_test.hdf5")

            self.box_feats_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_test.hdf5")

            self.human_pose_feat_test = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_test_bbox.hdf5")

            self.hoi_cands_union_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_test.hdf5")

            self.human_new_features_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_test.hdf5")
            self.object_new_features_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_test.hdf5")

            self.global_ids = self.load_subset_ids("test")
            print(len(self.global_ids))

        self.hoi_234_to_44 = io.load_json_object("data/vcoco/annotations/hoi_id_to_cluster_idx_45.json")
        assert len(self.hoi_234_to_44) == 234

        self.hoi_to_vec_numpy = np.load("data/vcoco/annotations/vcoco_hoi_600.npy")
        assert len(self.hoi_to_vec_numpy) == 234
        assert len(self.hoi_to_vec_numpy[0]) == 600

        self.hoi_dict = self.get_hoi_dict("data/vcoco/annotations/hoi_list_234.json")
        assert len(self.hoi_dict) == 234

        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids()

        self.table_list = io.load_json_object("data/vcoco/annotations/table_list.json")

        self.obj_to_id = self.get_obj_to_id("data/vcoco/annotations/object_list_80.json")
        assert len(self.obj_to_id) == 80

        self.verb_to_id = self.get_verb_to_id("data/vcoco/annotations/verb_list_25.json")
        assert len(self.verb_to_id) == 25

    def get_anno_dict(self, anno_list_json):
        anno_list = io.load_json_object(anno_list_json)
        anno_dict = {anno['global_id']: anno for anno in anno_list}
        return anno_dict

    def load_hdf5_file(self, hdf5_filename, mode='r'):
        return h5py.File(hdf5_filename, mode)

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {str(hoi['id']).zfill(3): hoi for hoi in hoi_list}
        return hoi_dict

    def get_obj_to_id(self, object_list_json):
        object_list = io.load_json_object(object_list_json)
        obj_to_id = {obj['name']: obj['id'] for obj in object_list}
        return obj_to_id

    def get_verb_to_id(self, verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['verb'] + "_" + verb['role']: verb['id'] for verb in verb_list}
        return verb_to_id

    def get_obj_to_hoi_ids(self):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in self.hoi_dict.items():
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]
        return obj_to_hoi_ids

    def load_subset_ids(self, subset):
        split_ids = io.load_json_object("data/vcoco/annotations/split_ids.json")
        subset_list_ = split_ids[subset]
        subset_list = []
        for id in subset_list_:
            subset_list.append(subset + "_" + id)
        return sorted(subset_list)

    def __len__(self):
        return len(self.global_ids)

    def get_labels(self, global_id):
        # hoi_idx: number in [0,29]
        if self.subset == "trainval":
            if global_id in self.hoi_cands_train.keys():
                hoi_cands = self.hoi_cands_train[global_id]
                hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]
                hoi_idxs = hoi_idxs.astype(np.int)
                # label: 0/1 indicating if there was a match with any gt for that hoi
                labels = self.hoi_cand_labels_train[global_id][()]

                num_cand = labels.shape[0]
                hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
                hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

                hoi_ids = [None] * num_cand
                for i in range(num_cand):
                    hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
                return hoi_ids, labels, hoi_label_vecs
            else:
                hoi_cands = self.hoi_cands_val[global_id]
                hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]
                hoi_idxs = hoi_idxs.astype(np.int)
                # label: 0/1 indicating if there was a match with any gt for that hoi
                labels = self.hoi_cand_labels_val[global_id][()]

                num_cand = labels.shape[0]
                hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
                hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

                hoi_ids = [None] * num_cand
                for i in range(num_cand):
                    hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
                return hoi_ids, labels, hoi_label_vecs
        else:
            hoi_cands = self.hoi_cands_test[global_id]
            hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]

            hoi_idxs = hoi_idxs.astype(np.int)
            # label: 0/1 indicating if there was a match with any gt for that hoi
            labels = self.hoi_cand_labels_test[global_id][()]

            num_cand = labels.shape[0]
            hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
            hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

            hoi_ids = [None] * num_cand
            for i in range(num_cand):
                hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
            return hoi_ids, labels, hoi_label_vecs

    def get_faster_rcnn_prob_vecs(self, hoi_ids, human_probs, object_probs):
        num_hois = len(self.hoi_dict)
        num_cand = len(hoi_ids)
        human_prob_vecs = np.tile(np.expand_dims(human_probs, 1), [1, num_hois])

        object_prob_vecs = np.zeros([num_cand, num_hois])
        for i, hoi_id in enumerate(hoi_ids):
            obj = self.hoi_dict[hoi_id]['object']
            obj_hoi_ids = self.obj_to_hoi_ids[obj]
            for obj_hoi_id in obj_hoi_ids:
                object_prob_vecs[i, int(obj_hoi_id) - 1] = object_probs[i]
        return human_prob_vecs, object_prob_vecs

    def sample_cands(self, hoi_labels):
        num_cands = hoi_labels.shape[0]
        indices = np.arange(num_cands)
        tp_ids = indices[hoi_labels == 1.0]
        fp_ids = indices[hoi_labels == 0]
        num_tp = tp_ids.shape[0]
        num_fp = fp_ids.shape[0]
        if num_tp == 0:
            num_fp_to_sample = self.fp_to_tp_ratio
        else:
            num_fp_to_sample = min(num_fp, self.fp_to_tp_ratio * num_tp)
        sampled_fp_ids = np.random.permutation(fp_ids)[:num_fp_to_sample]
        sampled_ids = np.concatenate((tp_ids, sampled_fp_ids), 0)
        return sampled_ids

    def get_obj_one_hot(self, hoi_ids):
        num_cand = len(hoi_ids)
        assert len(self.obj_to_id) == 80
        obj_one_hot = np.zeros([num_cand, len(self.obj_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            obj_id = self.obj_to_id[self.hoi_dict[hoi_id]['object']]
            obj_idx = int(obj_id) - 1
            obj_one_hot[i, obj_idx] = 1.0
        return obj_one_hot

    def get_verb_one_hot(self, hoi_ids):
        num_cand = len(hoi_ids)
        verb_one_hot = np.zeros([num_cand, len(self.verb_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            action = self.hoi_dict[hoi_id]['verb']
            object = self.hoi_dict[hoi_id]['object']

            d = self.table_list[action]

            verb_name = ""
            if "obj" in d and object in d["obj"]:
                verb_name = action + "_" + "obj"
            elif "instr" in d and object in d["instr"]:
                verb_name = action + "_" + "instr"
            assert verb_name != ""

            verb_id = self.verb_to_id[verb_name]
            verb_idx = int(verb_id) - 1
            verb_one_hot[i, verb_idx] = 1.0
        return verb_one_hot

    def get_prob_mask(self, hoi_idx):
        num_cand = len(hoi_idx)
        prob_mask = np.zeros([num_cand, len(self.hoi_dict)])
        prob_mask[np.arange(num_cand), hoi_idx] = 1.0
        return prob_mask

    def get_features(self, file, global_id, key="features"):
        features = file[global_id][key][()]
        features_indexs = file[global_id]['indexs'][()].tolist()
        ans = []
        for index in features_indexs:
            ans.append(features[int(index)])
        return np.array(ans)

    def get_verb_role_vec_list(self, hoi_idx):
        num = len(hoi_idx)
        vec_list = np.zeros([num, 600])
        for i, index in enumerate(hoi_idx):
            assert len(self.hoi_to_vec_numpy[index]) == 600
            vec_list[i] = self.hoi_to_vec_numpy[index, :]
        return vec_list

    def __getitem__(self, i):
        global_id = self.global_ids[i]

        if self.subset == "trainval":

            if global_id in self.hoi_cands_train.keys():
                start_end_ids = self.hoi_cands_train[global_id]['start_end_ids'][()]
                assert len(start_end_ids) == 234
                hoi_cands_ = self.hoi_cands_train[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
                hoi_ids_, hoi_labels_, hoi_label_vecs_, = self.get_labels(global_id)

                box_feats_ = self.box_feats_train[global_id][()]
                absolute_pose_feat_ = self.human_pose_feat_train[global_id]['absolute_pose'][()]
                relative_pose_feat_ = self.human_pose_feat_train[global_id]['relative_pose'][()]
                nis_labels_ = self.hoi_cand_nis_train[global_id][()]

                verb_obj_vec_ = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))
                human_features_ = self.get_features(self.human_new_features_train, global_id)
                object_features_ = self.get_features(self.object_new_features_train, global_id)
                union_features_ = self.get_features(self.hoi_cands_union_train, global_id)

                cand_ids = self.sample_cands(hoi_labels_)

                human_features = human_features_[cand_ids]
                object_features = object_features_[cand_ids]
                verb_obj_vec = verb_obj_vec_[cand_ids]
                hoi_cands = hoi_cands_[cand_ids]
                union_features = union_features_[cand_ids]
                hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
                hoi_labels = hoi_labels_[cand_ids]
                hoi_label_vecs = hoi_label_vecs_[cand_ids]
                box_feats = box_feats_[cand_ids]
                absolute_pose_feat = absolute_pose_feat_[cand_ids]
                relative_pose_feat = relative_pose_feat_[cand_ids]
                nis_labels = nis_labels_[cand_ids]

            else:
                start_end_ids = self.hoi_cands_val[global_id]['start_end_ids'][()]
                hoi_cands_ = self.hoi_cands_val[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
                hoi_ids_, hoi_labels_, hoi_label_vecs_ = self.get_labels(global_id)

                box_feats_ = self.box_feats_val[global_id][()]

                absolute_pose_feat_ = self.human_pose_feat_val[global_id]['absolute_pose'][()]
                relative_pose_feat_ = self.human_pose_feat_val[global_id]['relative_pose'][()]
                nis_labels_ = self.hoi_cand_nis_val[global_id][()]

                verb_obj_vec_ = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))
                human_features_ = self.get_features(self.human_new_features_val, global_id)
                object_features_ = self.get_features(self.object_new_features_val, global_id)
                union_features_ = self.get_features(self.hoi_cands_union_val, global_id)

                cand_ids = self.sample_cands(hoi_labels_)

                human_features = human_features_[cand_ids]
                object_features = object_features_[cand_ids]
                verb_obj_vec = verb_obj_vec_[cand_ids]
                hoi_cands = hoi_cands_[cand_ids]
                union_features = union_features_[cand_ids]
                hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
                hoi_labels = hoi_labels_[cand_ids]
                hoi_label_vecs = hoi_label_vecs_[cand_ids]
                box_feats = box_feats_[cand_ids]
                nis_labels = nis_labels_[cand_ids]

                absolute_pose_feat = absolute_pose_feat_[cand_ids]
                relative_pose_feat = relative_pose_feat_[cand_ids]

        else:
            start_end_ids = self.hoi_cands_test[global_id]['start_end_ids'][()]
            hoi_cands_ = self.hoi_cands_test[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
            hoi_ids, hoi_labels, hoi_label_vecs = self.get_labels(global_id)
            nis_labels = self.hoi_cand_nis_test[global_id][()]

            box_feats = self.box_feats_test[global_id][()]

            absolute_pose_feat = self.human_pose_feat_test[global_id]['absolute_pose'][()]
            relative_pose_feat = self.human_pose_feat_test[global_id]['relative_pose'][()]

            hoi_cands = hoi_cands_
            verb_obj_vec = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))

            human_features = self.get_features(self.human_new_features_test, global_id)
            object_features = self.get_features(self.object_new_features_test, global_id)
            union_features = self.get_features(self.hoi_cands_union_test, global_id)

        to_return = {
            'global_id': global_id,
            'human_box': hoi_cands[:, :4],
            'object_box': hoi_cands[:, 4:8],
            'human_prob': hoi_cands[:, 8],
            'object_prob': hoi_cands[:, 9],
            'human_rpn_id': hoi_cands[:, 10].astype(np.int),
            'object_rpn_id': hoi_cands[:, 11].astype(np.int),
            'hoi_idx': hoi_cands[:, -1].astype(np.int),
            'hoi_id': hoi_ids,
            'hoi_label': hoi_labels,
            'hoi_label_vec': hoi_label_vecs,
            'box_feat': box_feats,
            'absolute_pose': absolute_pose_feat,
            'relative_pose': relative_pose_feat,
            'hoi_cands_': hoi_cands_,
            'start_end_ids_': start_end_ids.astype(np.int),
            "union_features": union_features,
            "verb_obj_vec": verb_obj_vec,
            "human_feat": human_features,
            "object_feat": object_features,
            "nis_labels": nis_labels
        }

        human_prob_vecs, object_prob_vecs = self.get_faster_rcnn_prob_vecs(
            to_return['hoi_id'],
            to_return['human_prob'],
            to_return['object_prob'])

        to_return['human_prob_vec'] = human_prob_vecs
        to_return['object_prob_vec'] = object_prob_vecs

        to_return['object_one_hot'] = self.get_obj_one_hot(to_return['hoi_id'])
        to_return['verb_one_hot'] = self.get_verb_one_hot(to_return['hoi_id'])
        to_return['prob_mask'] = self.get_prob_mask(to_return['hoi_idx'])

        return to_return






