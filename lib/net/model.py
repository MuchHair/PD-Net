import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from net.layers import MLP, FactorAttentionTwoLevel
from utils import io as io


class HoiClassifier(nn.Module):
    def __init__(self, cluster_num):
        super(HoiClassifier, self).__init__()
        self.cluster_num = cluster_num
        self._init_mlp()

    def _init_mlp(self):
        cluster_num = self.cluster_num

        setattr(self, "sp_fc1", nn.Linear(2 * 21, 2 * 21))
        setattr(self, "sp_bn1", nn.BatchNorm1d(2 * 21))
        setattr(self, "sp_act1", nn.ReLU(inplace=True))
        setattr(self, "sp_fc2", nn.Linear(2 * 21, 2 * 21))
        setattr(self, "sp_bn2", nn.BatchNorm1d(2 * 21))
        setattr(self, "sp_act2", nn.ReLU(inplace=True))
        setattr(self, "sp_fc3", nn.Linear(2 * 21, cluster_num))

        setattr(self, "pose_fc1", nn.Linear(2 * (51 + 85), 2 * (51 + 85)))
        setattr(self, "pose_bn1", nn.BatchNorm1d(2 * (51 + 85)))
        setattr(self, "pose_act1", nn.ReLU(inplace=True))
        setattr(self, "pose_fc2", nn.Linear(2 * (51 + 85), 2 * (51 + 85)))
        setattr(self, "pose_bn2", nn.BatchNorm1d(2 * (51 + 85)))
        setattr(self, "pose_act2", nn.ReLU(inplace=True))
        setattr(self, "pose_fc3", nn.Linear(2 * (51 + 85), cluster_num))

        setattr(self, "human_fc1", nn.Linear(2048, 512))
        setattr(self, "human_bn1", nn.BatchNorm1d(512))
        setattr(self, "human_act1", nn.ReLU(inplace=True))
        setattr(self, "human_fc2", nn.Linear(512, cluster_num))

        setattr(self, "object_fc1", nn.Linear(2048, 512))
        setattr(self, "object_bn1", nn.BatchNorm1d(512))
        setattr(self, "object_act1", nn.ReLU(inplace=True))
        setattr(self, "object_fc2", nn.Linear(512, cluster_num))

        setattr(self, "union_fc1", nn.Linear(2048, 512))
        setattr(self, "union_bn1", nn.BatchNorm1d(512))
        setattr(self, "union_act1", nn.ReLU(inplace=True))
        setattr(self, "union_fc2", nn.Linear(512, cluster_num))

        setattr(self, "final_sigmoid", nn.Sigmoid())

    def transform_feat(self, feat):
        log_feat = torch.log(torch.abs(feat) + 1e-6)
        transformed_feat = torch.cat((feat, log_feat), 1)
        return transformed_feat

    def forward(self, features):
        transformed_sp_feats = self.transform_feat(features['box'])
        sp_feats = getattr(self, "sp_fc1")(transformed_sp_feats)
        sp_feats = getattr(self, "sp_bn1")(sp_feats)
        sp_feats = getattr(self, "sp_act1")(sp_feats)
        sp_feats = getattr(self, "sp_fc2")(sp_feats)
        sp_feats = getattr(self, "sp_bn2")(sp_feats)
        sp_feats = getattr(self, "sp_act2")(sp_feats)
        sp_feats = getattr(self, "sp_fc3")(sp_feats)

        human_feats = features['human_rcnn']
        human_feats = getattr(self, "human_fc1")(human_feats)
        human_feats = getattr(self, "human_bn1")(human_feats)
        human_feats = getattr(self, "human_act1")(human_feats)
        human_feats = getattr(self, "human_fc2")(human_feats)

        object_feats = features['object_rcnn']
        object_feats = getattr(self, "object_fc1")(object_feats)
        object_feats = getattr(self, "object_bn1")(object_feats)
        object_feats = getattr(self, "object_act1")(object_feats)
        object_feats = getattr(self, "object_fc2")(object_feats)

        absolute_pose = features['absolute_pose']
        relative_pose = features['relative_pose']
        pose_feats = torch.cat((absolute_pose, relative_pose), 1)
        transformed_pose_feats = self.transform_feat(pose_feats)
        pose_feats = getattr(self, "pose_fc1")(transformed_pose_feats)
        pose_feats = getattr(self, "pose_bn1")(pose_feats)
        pose_feats = getattr(self, "pose_act1")(pose_feats)
        pose_feats = getattr(self, "pose_fc2")(pose_feats)
        pose_feats = getattr(self, "pose_bn2")(pose_feats)
        pose_feats = getattr(self, "pose_act2")(pose_feats)
        pose_feats = getattr(self, "pose_fc3")(pose_feats)

        union_feats = features['union_rcnn']
        union_feats = getattr(self, "union_fc1")(union_feats)
        union_feats = getattr(self, "union_bn1")(union_feats)
        union_feats = getattr(self, "union_act1")(union_feats)
        union_feats = getattr(self, "union_fc2")(union_feats)

        factor_scores = 0
        factor_scores += sp_feats
        factor_scores += pose_feats
        factor_scores += human_feats
        factor_scores += object_feats
        factor_scores += union_feats

        verb_prob = getattr(self, "final_sigmoid")(factor_scores)
        verb_prob = ScatterVerbsToHois_234()(verb_prob)

        ans = verb_prob * features['human_prob_vec'] * features['object_prob_vec'] * features['prob_mask']

        return ans


class PD_Net(nn.Module):

    def __init__(self, use_pam=True, pam_num=4, is_hico=False):
        super(PD_Net, self).__init__()
        self.use_pam = use_pam
        self.pam_num = pam_num
        self.is_hico = is_hico
        if is_hico:
            hoi_num = 600
            verb_num = 117
        else:
            hoi_num = 234
            verb_num = 25
        # subejct channel
        self.SubjectSharedLayer = MLP(in_channel=2048, out_channel_list=[2048], activation_list=[True], bn_list=[True],
                                      drop_out_list=[False])
        self.SubjectVerbCls = nn.Linear(2048, verb_num)
        mid = int(math.sqrt(2048 / hoi_num) * hoi_num)
        self.SubjectHoiBlock = nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU(),
                                             MLP(in_channel=2048, out_channel_list=[mid, hoi_num],
                                                 bn_list=[True, False],
                                                 activation_list=[True, False], drop_out_list=[False] * 2))

        # object channel
        self.ObjectSharedLayer = MLP(in_channel=2048, out_channel_list=[2048], activation_list=[True], bn_list=[True],
                                     drop_out_list=[False])
        self.ObjectVerbCls = nn.Linear(2048, verb_num)
        mid = int(math.sqrt(2048 / hoi_num) * hoi_num)
        self.ObjectHoiBlock = nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU(),
                                            MLP(in_channel=2048, out_channel_list=[mid, hoi_num], bn_list=[True, False],
                                                activation_list=[True, False], drop_out_list=[False] * 2))

        # union channel
        self.UnionSharedLayer = MLP(in_channel=2048, out_channel_list=[2048], activation_list=[True], bn_list=[True],
                                    drop_out_list=[False])
        self.UnionVerbCls = nn.Linear(2048, verb_num)
        mid = int(math.sqrt(2048 / hoi_num) * hoi_num)
        self.UnionHoiBlock = nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU(),
                                           MLP(in_channel=2048, out_channel_list=[mid, hoi_num], bn_list=[True, False],
                                               activation_list=[True, False], drop_out_list=[False] * 2))

        spatial_feats_dim = 642
        pose_feats_dim = 600 + (136) * 2
        # spatial channel
        self.SpatialSharedLayer = MLP(in_channel=spatial_feats_dim,
                                      out_channel_list=[spatial_feats_dim, spatial_feats_dim],
                                      activation_list=[True, True],
                                      bn_list=[True, True], drop_out_list=[False] * 2)
        self.SpatialVerbCls = nn.Linear(spatial_feats_dim, verb_num)
        mid = int(math.sqrt(spatial_feats_dim / hoi_num) * hoi_num)
        self.SpatialHoiBlock = nn.Sequential(nn.BatchNorm1d(spatial_feats_dim), nn.ReLU(),
                                             MLP(in_channel=spatial_feats_dim, out_channel_list=[mid, hoi_num],
                                                 bn_list=[True, False], activation_list=[True, False],
                                                 drop_out_list=[False] * 2))

        # pose channel
        self.PoseSharedLayer = MLP(in_channel=pose_feats_dim, out_channel_list=[pose_feats_dim, pose_feats_dim],
                                   activation_list=[True] * 2, bn_list=[True] * 2, drop_out_list=[False] * 2)
        self.PoseVerbCls = nn.Linear(pose_feats_dim, verb_num)
        mid = int(math.sqrt(pose_feats_dim / hoi_num) * hoi_num)
        self.PoseHoiBlock = nn.Sequential(nn.BatchNorm1d(pose_feats_dim), nn.ReLU(),
                                          MLP(in_channel=pose_feats_dim, out_channel_list=[mid, hoi_num],
                                              bn_list=[True, False], activation_list=[True, False],
                                              drop_out_list=[False] * 2))

        if self.use_pam:
            self.factor_attention = FactorAttentionTwoLevel(600, self.pam_num)

    def transform_feat(self, feat):
        log_feat = torch.log(torch.abs(feat) + 1e-6)
        transformed_feat = torch.cat((feat, log_feat), 1)
        return transformed_feat

    def forward(self, feats):
        if self.use_pam:
            verbattention_score, hoi_attention_score = self.factor_attention(feats["verb_object_vec"])

        human_feats = self.SubjectSharedLayer(feats['human_feats'])
        obj_feats = self.ObjectSharedLayer(feats['object_feats'])
        union_feats = self.UnionSharedLayer(feats["union_feats"])

        transformed_sp_feats = self.transform_feat(feats['box'])
        spatial_feats = self.SpatialSharedLayer(torch.cat((transformed_sp_feats, feats["verb_object_vec"]), 1))

        absolute_pose = feats['absolute_pose']
        relative_pose = feats['relative_pose']
        pose_feats = torch.cat((absolute_pose, relative_pose), 1)
        transformed_pose_feats = self.transform_feat(pose_feats)
        pose_feats_ = torch.cat((transformed_pose_feats, feats["verb_object_vec"]), 1)
        pose_feats = self.PoseSharedLayer(pose_feats_)

        verb_scores = {}
        verb_scores['human_channel'] = self.SubjectVerbCls(human_feats)
        verb_scores['object_channel'] = self.ObjectVerbCls(obj_feats)
        verb_scores['spatial_channel'] = self.SpatialVerbCls(spatial_feats)
        verb_scores['pose_channel'] = self.PoseVerbCls(pose_feats)
        verb_scores['union_channel'] = self.UnionVerbCls(union_feats)

        verb_scores['all_channel'] = 0

        # hoi scores
        hoi_scores = {}
        hoi_scores['human_channel'] = self.SubjectHoiBlock(human_feats)
        hoi_scores['object_channel'] = self.ObjectHoiBlock(obj_feats)
        hoi_scores['spatial_channel'] = self.SpatialHoiBlock(spatial_feats)
        hoi_scores['pose_channel'] = self.PoseHoiBlock(pose_feats)
        hoi_scores['union_channel'] = self.UnionHoiBlock(union_feats)

        hoi_scores['all_channel'] = 0

        if self.use_pam:
            for j, feat_name in zip(range(self.pam_num), verb_scores.keys()):
                # print(feat_name)
                verbattention_score_j = verbattention_score[:, j].contiguous()
                verbattention_score_j = verbattention_score_j.view(verbattention_score_j.shape[0], -1)
                verb_scores['all_channel'] += verbattention_score_j * verb_scores[feat_name]

                hoi_attention_score_j = hoi_attention_score[:, j].contiguous()
                hoi_attention_score_j = hoi_attention_score_j.view(verbattention_score_j.shape[0], -1)
                hoi_scores['all_channel'] += hoi_attention_score_j * hoi_scores[feat_name]

            if self.pam_num == 4 and not self.is_hico:
                verb_scores['all_channel'] += verb_scores["union_channel"]
                hoi_scores['all_channel'] += hoi_scores["union_channel"]

        verb_cls = nn.Sigmoid()(verb_scores['all_channel'])
        if self.is_hico:
            verb_cls = ScatterVerbsToHois_600()(verb_cls)
        else:
            verb_cls = ScatterVerbsToHois_234()(verb_cls)
        verb_cls = verb_cls * feats['human_prob_vec'] * feats['object_prob_vec']
        verb_cls = verb_cls * feats['prob_mask']

        hoi_cls = nn.Sigmoid()(hoi_scores['all_channel'])
        hoi_cls = hoi_cls * feats['human_prob_vec'] * feats['object_prob_vec']
        hoi_cls = hoi_cls * feats['prob_mask']
        return verb_cls, hoi_cls


class ScatterVerbsToHois_234(nn.Module):
    def __init__(self):
        super(ScatterVerbsToHois_234, self).__init__()
        self.hoi_dict = self.get_hoi_dict('data/vcoco/annotations/hoi_list_234.json')
        self.verb_to_id = self.get_verb_to_id('data/vcoco/annotations/verb_list_25.json')

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def get_verb_to_id(self, verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['verb'] + "_" + verb['role']: verb['id'] for verb in verb_list}
        return verb_to_id

    def forward(self, verb_scores):
        batch_size, num_verbs = verb_scores.size()
        num_hois = len(self.hoi_dict)
        hoi_scores = Variable(torch.zeros(batch_size, num_hois)).cuda()
        for hoi_id, hoi in self.hoi_dict.items():
            action = hoi['verb']
            role = hoi['role']
            verb_name = action + "_" + role

            verb_idx = int(self.verb_to_id[verb_name]) - 1
            hoi_idx = int(hoi_id) - 1
            hoi_scores[:, hoi_idx] = verb_scores[:, verb_idx]
        return hoi_scores


class ScatterVerbsToHois_600(nn.Module):
    def __init__(self):
        super(ScatterVerbsToHois_600, self).__init__()
        self.hoi_dict = self.get_hoi_dict('data/vcoco/annotations/hoi_list.json')
        self.verb_to_id = self.get_verb_to_id('data/vcoco/annotations/verb_list.json')

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def get_verb_to_id(self, verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['name']: verb['id'] for verb in verb_list}
        return verb_to_id

    def forward(self, verb_scores):
        batch_size, num_verbs = verb_scores.size()
        num_hois = len(self.hoi_dict)
        hoi_scores = Variable(torch.zeros(batch_size, num_hois)).cuda()
        for hoi_id, hoi in self.hoi_dict.items():
            verb = hoi['verb']
            verb_idx = int(self.verb_to_id[verb]) - 1
            hoi_idx = int(hoi_id) - 1
            hoi_scores[:, hoi_idx] = verb_scores[:, verb_idx]
        return hoi_scores


class HoiClassifierNIS(nn.Module):
    def __init__(self):
        super(HoiClassifierNIS, self).__init__()
        self._init_mlp()

    def _init_mlp(self):
        setattr(self, "sp_fc1", nn.Linear(2 * 21, 500))
        setattr(self, "sp_act1", nn.ReLU(inplace=True))

        setattr(self, "human_fc1", nn.Linear(2048, 500))
        setattr(self, "human_act1", nn.ReLU(inplace=True))

        setattr(self, "object_fc1", nn.Linear(2048, 500))
        setattr(self, "object_act1", nn.ReLU(inplace=True))

        setattr(self, "object_fc1", nn.Linear(2048, 500))
        setattr(self, "object_act1", nn.ReLU(inplace=True))

        setattr(self, "final_fc1", nn.Linear(1500, 1000))
        setattr(self, "final_relu", nn.ReLU(inplace=True))

        setattr(self, "final_fc2", nn.Linear(1000, 1))
        setattr(self, "final_sigmoid", nn.Sigmoid())

    def transform_feat(self, feat):
        log_feat = torch.log(torch.abs(feat) + 1e-6)
        transformed_feat = torch.cat((feat, log_feat), 1)
        return transformed_feat

    def forward(self, features):
        sp_feats = self.transform_feat(features['box'])
        sp_feats = getattr(self, "sp_fc1")(sp_feats)
        sp_feats = getattr(self, "sp_act1")(sp_feats)

        human_feats = features['human_rcnn']
        human_feats = getattr(self, "human_fc1")(human_feats)
        human_feats = getattr(self, "human_act1")(human_feats)

        object_feats = features['object_rcnn']
        object_feats = getattr(self, "object_fc1")(object_feats)
        object_feats = getattr(self, "object_act1")(object_feats)

        all_feats = torch.cat((human_feats, object_feats, sp_feats), 1)
        all_feats = getattr(self, "final_fc1")(all_feats)
        all_feats = getattr(self, "final_relu")(all_feats)
        all_feats = getattr(self, "final_fc2")(all_feats)
        binary_prob = getattr(self, "final_sigmoid")(all_feats)

        return binary_prob
