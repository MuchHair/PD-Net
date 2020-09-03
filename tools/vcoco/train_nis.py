import sys
sys.path.insert(0, "lib")

from net.model import HoiClassifierNIS

from dataset.dataset import Features_PD_VCOCO
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

import utils.io as io


def train_model(model, dataset_train, dataset_val, lr, num_epochs, model_dir, exp_name, scale_lr=None):
    params = itertools.chain(
        model.parameters())
    optimizer = optim.Adam(params, lr=lr)

    criterion = nn.BCELoss()

    step = 0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            feats = {
                'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
                'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
                'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
                "human_det_score": Variable(torch.cuda.FloatTensor(data["human_prob"])),
                "object_det_score": Variable(torch.cuda.FloatTensor(data["object_prob"])),
                "object_word2vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec'][:, 300:])),
            }

            model.train()
            binary_score = model(feats)

            # add
            binary_label = Variable(torch.cuda.FloatTensor(data['nis_labels']))
            loss_binary = criterion(binary_score, binary_label.view(binary_score.size(0), 1))

            loss_binary.backward()
            if step % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 20 == 0:
                num_tp = np.sum(data['hoi_label'])
                num_fp = data['hoi_label'].shape[0] - num_tp
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    ' Train Loss binary: {:.8f}' \
                    '| TPs: {} | FPs: {} | lr:{} '
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss_binary.data[0],
                    num_tp,
                    num_fp,
                    optimizer.param_groups[0]['lr'])
                print(log_str)

            if step % 100 == 0:
                log_value('train_loss_binary', loss_binary.data[0], step)
                print(exp_name)

            if step % 1000 == 0 and step > 2000:
                val_loss_binary, recall, tp, fp = eval_model(model, dataset_val)
                log_value('val_loss_binary', val_loss_binary, step)
                log_value('recall', recall, step)
                log_value('tp', tp, step)
                log_value('fp', fp, step)

                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss binary: {:.8f}' \
                    '| recall: {:.2f} | tp: {:.2f}|fp: {:.2f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    val_loss_binary,
                    recall,
                    tp,
                    fp)
                print(log_str)

            if step == 10 or( step % 1000 == 0 and step > 2000):
                hoi_classifier_pth = os.path.join(
                    model_dir, "model",
                    f'hoi_classifier_{step}')
                torch.save(
                    model.state_dict(),
                    hoi_classifier_pth)

            step += 1

            if scale_lr is not None and step == scale_lr:
                scale_lr(optimizer, 0.1)


def eval_model(model, dataset):
    model.eval()
    criterion = nn.BCELoss()
    step = 0
    val_loss_binary = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)

    total_p = 0
    p_4 = 0
    f_p_4 = 0
    total_predict = 0

    for sample_id in tqdm(sampler):

        data = dataset[sample_id]
        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
            'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
            "human_det_score": Variable(torch.cuda.FloatTensor(data["human_prob"])),
            "object_det_score": Variable(torch.cuda.FloatTensor(data["object_prob"])),
            "object_word2vec": Variable(torch.cuda.FloatTensor(data['verb_obj_vec'][:, 300:])),
        }

        binary_score = model(feats)
        binary_label = Variable(torch.cuda.FloatTensor(data['nis_labels']))
        batch_size = binary_score.size(0)

        loss_binary = criterion(binary_score, binary_label.view(batch_size, 1))

        for i in range(len(binary_label)):
            if data['nis_labels'][i] == 1:
                total_p = total_p + 1
                if binary_score.data[i].cpu().numpy() > 0.8:
                    total_predict = total_predict + 1
                    p_4 = p_4 + 1
                if binary_score.data[i].cpu().numpy() > 0.8:
                    total_predict = total_predict + 1
                    f_p_4 = f_p_4 + 1


        val_loss_binary += (batch_size * loss_binary.data[0])

        count += batch_size
        step += 1

    val_loss_binary = val_loss_binary / float(count)
    print(total_p, p_4 / total_p)

    if total_predict > 0:
        return val_loss_binary, p_4 / total_p, p_4 / total_predict, f_p_4 / total_predict
    else:
        return val_loss_binary, p_4 / total_p, 0, 0


def main():
    model_dir = "output/vcoco/nis/"
    print(model_dir)
    lr = 1e-4
    num_epochs = 10
    io.mkdir_if_not_exists(model_dir, recursive=True)
    io.mkdir_if_not_exists(os.path.join(model_dir, "model"))
    io.mkdir_if_not_exists(os.path.join(model_dir, "log"))
    configure(os.path.join(model_dir, "log"))

    model = HoiClassifierNIS().cuda()

    print('Creating data loaders ...')
    dataset_train = Features_PD_VCOCO(subset="trainval")
    dataset_train.fp_to_tp_ratio = 3
    dataset_val = Features_PD_VCOCO(subset="test")

    train_model(model, dataset_train, dataset_val, lr, num_epochs, model_dir, model_dir)


if __name__ == "__main__":
    main()

