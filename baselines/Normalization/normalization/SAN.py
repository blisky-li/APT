import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_


class SAN(nn.Module):
    def __init__(self, features='M',station_type='adaptive', **model_args):
        super(SAN, self).__init__()
        self.configs = model_args
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.period_len = model_args['period_len']
        self.features = features
        self.channels = model_args['enc_in'] if self.features == 'M' else 1
        self.station_type = station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):

        channels = self.channels
        seq_len = self.seq_len // self.period_len
        pred_len = self.pred_len_new
        period_len = self.period_len
        self.model = MLP(mode='mean', seq_len=seq_len, pred_len=pred_len, channels=channels, period_len=period_len).float()
        self.model_std = MLP(mode='std',  seq_len=seq_len, pred_len=pred_len, channels=channels, period_len=period_len).float()

    def normalize(self, input):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)
            # print(input.shape, mean.shape, outputs_mean.shape)
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)[:, -self.pred_len_new:, :]

            if self.features == 'MS':
                outputs = outputs[:, :, [self.channels - 1, -1]]
            # print(outputs.shape)
            self.station_pred = outputs

            return norm_input.reshape(bs, len, dim)

        else:
            return input, None

    def de_normalize(self, input):
        if self.station_type == 'adaptive':
            station_pred = self.station_pred
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)
        else:
            return input
    def forward(self, input, mode='norm'):
        if mode == 'norm':
            return self.normalize(input)
        elif mode == 'denorm':
            return self.de_normalize(input)
        else:
            return input

class MLP(nn.Module):
    def __init__(self, mode, seq_len, pred_len, channels, period_len):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
