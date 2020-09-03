import sys

sys.path.insert(0, "lib")

from net.model import HoiClassifier

import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler
from dataset.dataset import Features_PD_VCOCO
import utils.io as io


def train_model(model, dataset_train, dataset_val, lr, num_epochs, model_dir, exp_name):
    params = itertools.chain(
        model.parameters())
    optimizer = optim.Adam(params, lr=lr)

    criterion = nn.BCELoss()

    step = 0
    optimizer.zero_grad()
    for epoch in range(0, num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            feats = {
                'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
                'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
                'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
                'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose'])),
                'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose'])),
                'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
                'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
                'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot'])),
                'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask'])),
                'union_rcnn': Variable(torch.cuda.FloatTensor(data['union_features'])),
                "human_prob": Variable(torch.cuda.FloatTensor(data['human_prob'])),
                "object_prob": Variable(torch.cuda.FloatTensor(data['object_prob'])),
                "verb_obj_vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec'])),
            }

            model.train()
            hoi_prob = model(feats)
            hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))
            loss = criterion(hoi_prob, hoi_labels)
            loss.backward()
            if step % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

            max_prob = hoi_prob.max().data[0]
            max_prob_tp = torch.max(hoi_prob * hoi_labels).data[0]

            if step % 20 == 0:
                num_tp = np.sum(data['hoi_label'])
                num_fp = data['hoi_label'].shape[0] - num_tp
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Train Loss: {:.8f} | TPs: {} | FPs: {} | ' + \
                    'Max TP Prob: {:.8f} | Max Prob: {:.8f} | lr:{}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data[0],
                    num_tp,
                    num_fp,
                    max_prob_tp,
                    max_prob,
                    optimizer.param_groups[0]["lr"])
                print(log_str)

            if step % 100 == 0:
                log_value('train_loss', loss.data[0], step)
                log_value('max_prob', max_prob, step)
                log_value('max_prob_tp', max_prob_tp, step)
                print(exp_name)

            if step % 1000 == 0 and step > 9000:
                val_loss = eval_model(model, dataset_val)
                log_value('val_loss', val_loss, step)
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    val_loss)
                print(log_str)

            if step == 10 or (step % 1000 == 0 and step > 9000):
                hoi_classifier_pth = os.path.join(
                    model_dir, "model",
                    f'hoi_classifier_{step}')
                torch.save(
                    model.state_dict(),
                    hoi_classifier_pth)

            step += 1


def eval_model(model, dataset):
    model.eval()
    criterion = nn.BCELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
            'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
            'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose'])),
            'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose'])),
            'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
            'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
            'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot'])),
            'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask'])),
            'union_rcnn': Variable(torch.cuda.FloatTensor(data['union_features'])),
            "human_prob": Variable(torch.cuda.FloatTensor(data['human_prob'])),
            "object_prob": Variable(torch.cuda.FloatTensor(data['object_prob'])),
            "verb_obj_vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec'])),
        }
        hoi_prob = model(feats)

        hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))
        loss = criterion(hoi_prob, hoi_labels)

        batch_size = hoi_prob.size(0)
        val_loss += (batch_size * loss.data[0])
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss


def main_45_5_stream_152():
    model = HoiClassifier(25).cuda()
    lr = 1e-4
    num_epochs = 10
    model_dir = "output/vcoco/SH_5_stream_baseline_152"
    io.mkdir_if_not_exists(model_dir, recursive=True)
    io.mkdir_if_not_exists(os.path.join(model_dir, "log"))
    io.mkdir_if_not_exists(os.path.join(model_dir, "model"))

    configure(os.path.join(model_dir, "log"))
    dataset_train = Features_PD_VCOCO(subset="trainval", fp_to_tp_ratio=1000)
    dataset_val = Features_PD_VCOCO(subset="test")
    train_model(model, dataset_train, dataset_val, lr, num_epochs, model_dir, model_dir)


if __name__ == "__main__":
    main_45_5_stream_152()

