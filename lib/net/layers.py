import torch.nn as nn
import math


class FactorAttentionTwoLevel(nn.Module):

    def __init__(self, obj_channel, out_channel):
        super(FactorAttentionTwoLevel, self).__init__()

        self.fc1 = nn.Linear(obj_channel, obj_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(obj_channel, out_channel)
        mid_channel = int(math.sqrt(obj_channel / out_channel) * out_channel)
        self.final_layer = MLP(in_channel=obj_channel, out_channel_list=[mid_channel, out_channel], bn_list=[False] * 2,
                               activation_list=[True, False], drop_out_list=[False] * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out1 = self.sigmoid(self.fc2(out))
        out2 = self.sigmoid(self.final_layer(out))

        return out1, out2


class fc_block_layer(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True, bn=False, drop_out=False):
        super(fc_block_layer, self).__init__()

        self.layers = nn.ModuleList()
        fc = nn.Linear(in_channel, out_channel)
        self.layers.append(fc)
        if bn:
            self.layers.append(nn.BatchNorm1d(out_channel))
        if activation:
            self.layers.append(nn.ReLU())
        if drop_out:
            self.layers.append(nn.Dropout())

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel_list, activation_list, bn_list, drop_out_list):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, out_channel in enumerate(out_channel_list):
            activation = activation_list[i]
            bn = bn_list[i]
            drop_out = drop_out_list[i]
            fc_block = fc_block_layer(in_channel, out_channel, activation, bn, drop_out)
            self.layers.append(fc_block)
            in_channel = out_channel

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
