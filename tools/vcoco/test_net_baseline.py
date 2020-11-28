import os

import sys

sys.path.insert(0, "lib")
from dataset.dataset import Features_PD_VCOCO

import utils.io as io
import h5py
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import numpy as np

from net.model import HoiClassifier


def generate_pkl(pred_dets_hdf5, out_dir, file_name):
    print(pred_dets_hdf5)
    print(out_dir)
    print(file_name)

    pred_dets = h5py.File(pred_dets_hdf5, 'r')
    print(len(pred_dets.keys()))
    assert len(pred_dets.keys()) == 4539

    hoi_list = io.load_json_object("data_symlinks/vcoco_processed/new/hoi_list_234.json")
    hoi_dict = {int(hoi["id"]) - 1: hoi for hoi in hoi_list}

    result_list_all = []
    for global_id in tqdm(pred_dets.keys()):
        result_list = []
        image_id = int(global_id.split("_")[1])

        start_end_ids = pred_dets[global_id]['start_end_ids']
        assert len(start_end_ids) == 234

        for hoi_id in range(234):
            start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)]

            if start_id == end_id:
                continue

            hoi_dets_all = pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

            hoi_dets_parts = []
            for dets in hoi_dets_all:
                if hoi_dets_parts == []:
                    hoi_dets_parts.append(dets)
                else:
                    find = False
                    for i in range(len(hoi_dets_parts)):
                        origin_dets = hoi_dets_parts[i]
                        if origin_dets[0] == dets[0] and origin_dets[1] == dets[1] \
                                and origin_dets[2] == dets[2] and origin_dets[3] == dets[3]:
                            find = True
                            origin_score = origin_dets[8]
                            cur_score = dets[8]
                            if cur_score > origin_score:
                                hoi_dets_parts[i] = dets
                            break
                    if not find:
                        hoi_dets_parts.append(dets)

            for hoi_dets in hoi_dets_parts:
                person_boxes = hoi_dets[:4].tolist()
                aciton = hoi_dict[hoi_id]["verb"]
                role = hoi_dict[hoi_id]["role"]
                find = False
                for result in result_list:
                    if result["image_id"] == image_id and result["person_box"] == person_boxes:
                        if aciton + "_" + role not in result:
                            result[aciton + "_" + role] = [hoi_dets[4], hoi_dets[5],
                                                           hoi_dets[6], hoi_dets[7],
                                                           hoi_dets[8]]
                        else:
                            if hoi_dets[8] > result[aciton + "_" + role][4]:
                                result[aciton + "_" + role] = [hoi_dets[4], hoi_dets[5],
                                                               hoi_dets[6], hoi_dets[7],
                                                               hoi_dets[8]]
                        find = True
                        break
                if not find:
                    per_image_dict = {}
                    per_image_dict["image_id"] = image_id
                    per_image_dict["person_box"] = person_boxes
                    per_image_dict[aciton + "_" + role] = [hoi_dets[4], hoi_dets[5],
                                                           hoi_dets[6], hoi_dets[7],
                                                           hoi_dets[8]]

                    result_list.append(per_image_dict)
        result_list_all.extend(result_list)
    io.dump_pickle_object(result_list_all, os.path.join(out_dir, file_name + ".pkl"))


def eval_model(model, dataset, output_dir, model_num):
    print('Creating hdf5 file for predicted hoi dets ...')

    pred_hoi_dets_hdf5 = os.path.join(
        output_dir,
        f'pred_hoi_dets_test_{model_num}.hdf5')

    pred_hois = h5py.File(pred_hoi_dets_hdf5, 'w')
    model.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]

        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat']), volatile=True),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat']), volatile=True),
            'box': Variable(torch.cuda.FloatTensor(data['box_feat']), volatile=True),
            'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose']), volatile=True),
            'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose']), volatile=True),
            'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec']), volatile=True),
            'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec']), volatile=True),
            'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot']), volatile=True),
            'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask']), volatile=True),
            'union_rcnn': Variable(torch.cuda.FloatTensor(data['union_features']), volatile=True),
            "verb_obj_vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec']), volatile=True),
            "human_prob": Variable(torch.cuda.FloatTensor(data['human_prob']), volatile=True),
            "object_prob": Variable(torch.cuda.FloatTensor(data['object_prob']), volatile=True),
        }

        hoi_prob = model(feats)

        hoi_prob = hoi_prob.data.cpu().numpy()

        num_cand = hoi_prob.shape[0]
        scores = hoi_prob[np.arange(num_cand), np.array(data['cluster_idx'])]

        human_obj_boxes_scores = np.concatenate((
            data['human_box'],
            data['object_box'],
            np.expand_dims(scores, 1)), 1)

        global_id = data['global_id']
        pred_hois.create_group(global_id)
        pred_hois[global_id].create_dataset(
            'human_obj_boxes_scores',
            data=human_obj_boxes_scores)
        pred_hois[global_id].create_dataset(
            'start_end_ids',
            data=data['start_end_ids_'])

    pred_hois.close()


def main_45_5_stream_152():
    model_num = 19000
    model_dir = "SH_5_stream_baseline_152"
    print(model_dir, model_num)
    model_path = f"output/vcoco/{model_dir}/" \
                 f"model/hoi_classifier_{model_num}"
    output_dir = f"output/vcoco/{model_dir}"

    model = HoiClassifier(25).cuda()
    model.load_state_dict(torch.load(model_path))
    print("Creating data loader ...")
    dataset = Features_PD_VCOCO(subset="test")
    eval_model(model, dataset, output_dir, model_num)

    generate_pkl(
        f"output/vcoco/{model_dir}/"
        f"pred_hoi_dets_test_{model_num}.hdf5",
        f"output/vcoco/{model_dir}/",
        f"{model_num}")


if __name__ == "__main__":
    main_45_5_stream_152()