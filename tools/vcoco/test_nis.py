import os
import h5py
import torch
from tqdm import tqdm
import sys

sys.path.insert(0, "lib")
from dataset.dataset import Features_PD_VCOCO
from net.model import HoiClassifierNIS

tqdm.monitor_interval = 0
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler


def eval_model(model, dataset, model_num, output_dir):
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
            "human_det_score": Variable(torch.cuda.FloatTensor(data["human_prob"]), volatile=True),
            "object_det_score": Variable(torch.cuda.FloatTensor(data["object_prob"]), volatile=True),
            "object_word2vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec'][:, 300:]), volatile=True),
        }

        binary_score = model(feats)

        binary_score_data = binary_score.data.cpu().numpy()

        global_id = data['global_id']
        pred_hois.create_group(global_id)
        pred_hois[global_id].create_dataset(
            'binary_score_data',
            data=binary_score_data)
        pred_hois[global_id].create_dataset(
            'start_end_ids',
            data=data['start_end_ids_'])

    pred_hois.close()


def main():
    model_num = 39000
    model_path = "output/vcoco/nis/" \
                 f"model/hoi_classifier_{model_num}"
    output_dir = "output/vcoco/nis/"

    print('Loading model ...')
    print(model_path)
    print(model_num)
    model = HoiClassifierNIS().cuda()
    model.load_state_dict(torch.load(model_path))

    print('Creating data loader ...')
    dataset = Features_PD_VCOCO(subset="test")
    eval_model(model, dataset, model_num, output_dir)


if __name__ == "__main__":
    main()
