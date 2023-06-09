# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from Source.ChangePrediction.TrainConfig import *

#
# class Config(object):
#     def __init__(self, data_dir):
#         # self.model_name = 'mmoe'
#         # self.train_path = data_dir + 'census-income.data.gz'
#         # self.test_path = data_dir + 'census-income.test.gz'
#         # self.save_path = './saved_dict/' + self.model_name + '.ckpt'
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # self.require_improvement = 1000
#         self.dropout = 0.5
#         self.learning_rate = 3e-5
#         self.label_columns = target_labels
#
#         # self.label_dict = [2, 2]
#         self.num_feature = len(feature_list)
#         self.num_experts = 5
#         self.num_tasks = len(target_labels)
#         self.units = 16
#         self.hidden_units = 8
#         self.embed_size = 300
#         # self.batch_size = 256
#         # self.field_size = 0
#         self.towers_hidden = 16
#         self.SB_hidden = 1024
#         self.SB_output = 512
#         # self.num_epochs = 100
#         # self.loss_fn = loss_fn('binary')


class Transform_layer(nn.Module):
    def __init__(self, input_size, output_size, num_experts):
        super(Transform_layer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,), device=train_device), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 2

        w = torch.empty(input_size, num_experts, output_size, device=train_device)
        self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),
                                    requires_grad=True)

        w = torch.empty(input_size, num_experts, output_size, device=train_device)
        self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                           requires_grad=True)

    def forward(self, x):
        self.s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        self.s_ = self.s * (self.eplison - self.gamma) + self.gamma

        self.z_params = (self.s_ > 0).float() * self.s_
        self.z_params = (self.z_params > 1).float() + (self.z_params <= 1).float() * self.z_params

        output = self.z_params * self.w_params
        output = torch.einsum('ab,bnc -> anc', x, output)
        return output


class high_layers(nn.Module):

    def __init__(self, input_size, output_size):
        super(high_layers, self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,), device=train_device), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 2

        w = torch.empty(input_size, output_size, device=train_device)
        self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),
                                    requires_grad=True)

        w = torch.empty(input_size, output_size, device=train_device)
        self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                           requires_grad=True)

    def forward(self, x):
        self.s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        self.s_ = self.s * (self.eplison - self.gamma) + self.gamma

        self.z_params = (self.s_ > 0).float() * self.s_
        self.z_params = (self.z_params > 1).float() + (self.z_params <= 1).float() * self.z_params

        output = self.z_params * self.w_params
        output = torch.einsum('anc,cd -> and', x, output)
        return output


class Tower(nn.Module):
    def __init__(self, input_size, output_size, task_type, hidden_size=16):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.task_type = task_type

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.task_type == 'binary':
            out = torch.sigmoid(out)
        return out


class Model(nn.Module):
    def __init__(self, num_feature, SB_hidden, SB_output, num_experts, towers_hidden, num_tasks, task_types, task_out_dims, device):
        super(Model, self).__init__()

        # accept_unit = config.field_size*config.embed_size
        accept_unit = num_feature
        self.trans1 = Transform_layer(accept_unit, SB_hidden, num_experts)
        self.trans2 = high_layers(SB_hidden, SB_output)

        self.fc_experts = nn.Linear(num_experts, 1)
        self.relu = nn.ReLU()

        self.towers = nn.ModuleList([Tower(SB_output, task_out_dims[i], hidden_size=towers_hidden, task_type=task_types[i])
                                     for i in range(num_tasks)])

        self.lamdba = 1e-4

        # self.embedding_layer = nn.Embedding(config.num_feature,config.embed_size)

        self.to(device)

    def forward(self, x):
        output = self.trans1(x)
        output = self.trans2(output)
        output = output.transpose(2, 1)
        output = self.fc_experts(output)
        output = torch.squeeze(output)
        output = self.relu(output)

        final_outputs = [tower(output) for tower in self.towers]
        results = torch.cat(final_outputs, -1)

        s1 = self.trans1.s_
        s2 = self.trans2.s_

        s1_prob = 1 - ((s1 < 0).sum(dim=-1) / s1.size(1))
        s2_prob = 1 - ((s2 < 0).sum(dim=-1) / s2.size(1))

        regul = self.lamdba * (s1_prob.sum() + s2_prob.sum())
        # regul = 0
        # return results, regul
        return results
