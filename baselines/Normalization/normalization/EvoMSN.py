import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

"""
Using per instance per variate period weight
"""


class EvoMSN(nn.Module):
    def __init__(self, period_list, device, features='M', station_type='adaptive', **model_args):
        super(EvoMSN, self).__init__()
        self.device = device
        self.configs = configs
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.period_list = period_list
        self.k = model_args['top_k']
        self.features = features
        self.channels = model_args['enc_in'] if self.features == 'M' else 1
        self.station_type = station_type
        self.epsilon = 1e-5
        self._build_model()

    def _build_model(self):
        self.stat_predict = []
        for i in range(self.k):
            period_len = self.period_list[i]
            if self.seq_len % period_len != 0:
                seq_len = (self.seq_len // period_len) + 1
            else:
                seq_len = self.seq_len // period_len

            if self.pred_len % period_len != 0:
                pred_len = (self.pred_len // period_len) + 1
            else:
                pred_len = self.pred_len // period_len

            self.stat_predict.append(PredModule(seq_len=seq_len,
                                                pred_len=pred_len,
                                                features = self.features
                                                **model_args).to(self.device))

    def _get_weight(self, x):
        xf = torch.fft.rfft(x, dim=1)
        period_index = x.shape[1] // self.period_list
        amps = abs(xf)
        period_amps = amps[:, period_index, :]
        period_weight = period_amps / period_amps.sum(axis=1, keepdims=True)
        return period_weight

    def normalize(self, input):
        if 'adaptive' in self.station_type:
            bs, len, dim = input.shape
            normalized_input = []
            stat_pred = []
            for i in range(self.k):
                period_len = self.period_list[i]
                if len % period_len != 0:
                    length = ((len // period_len) + 1) * period_len
                    padding = input[:, -(length - len):, :]
                    input = torch.cat([input, padding], dim=1)
                input = input.reshape(bs, -1, period_len, dim)
                mean = torch.mean(input, dim=-2, keepdim=True)
                std = torch.std(input, dim=-2, keepdim=True)
                norm_input = (input - mean) / (std + self.epsilon)
                norm_input = norm_input.reshape(bs, -1, dim)[:, :len, :]
                normalized_input.append(norm_input)
                input = input.reshape(bs, -1, dim)[:, :len, :]
                stat_pred.append(self.stat_predict[i](input, mean, std))
            self.stat_pred = stat_pred
            return normalized_input
        else:
            return input, None

    def de_normalize(self, input_list, return_outputlist=False):
        if 'adaptive' in self.station_type:
            bs, len, dim = input_list[0].shape
            denormalized_output = []
            for i in range(self.k):
                input = input_list[i]
                period_len = self.period_list[i]
                if len % period_len != 0:
                    length = ((len // period_len) + 1) * period_len
                    padding = input[:, -(length - len):, :]
                    input = torch.cat([input, padding], dim=1)
                input = input.reshape(bs, -1, period_len, dim)
                mean = self.stat_pred[i][:, :, :dim].unsqueeze(2)
                std = self.stat_pred[i][:, :, dim:].unsqueeze(2)
                denorm_output = input * (std + self.epsilon) + mean
                denorm_output = denorm_output.reshape(bs, -1, dim)[:, :len, :]
                denormalized_output.append(denorm_output)
                input = input.reshape(bs, -1, dim)[:, :len, :]
            denormalized_output = torch.stack(denormalized_output, dim=-1)
            self.period_weight = self._get_weight(input).permute(0, 2, 1)
            weight = self.period_weight.unsqueeze(1).repeat(1, len, 1, 1).to(input.device)
            weighted_denorm_output = torch.sum(denormalized_output * weight, -1).reshape(bs, len, dim)
            if return_outputlist:
                return weighted_denorm_output.float(), denormalized_output
            else:
                return weighted_denorm_output.float()
        else:
            return input_list


class PredModule(nn.Module):
    def __init__(self, seq_len, pred_len, features='M', **model_args):
        super(PredModule, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = model_args['enc_in'] if features == 'M' else 1

        self.model_mean = MLP(input_len=model_args['seq_len'],
                              seq_len=seq_len,
                              pred_len=pred_len,
                              enc_in=self.channels,
                              mode='mean')
        self.model_std = MLP(input_len=model_args['seq_len'],
                             seq_len=seq_len,
                             pred_len=pred_len,
                             enc_in=self.channels,
                             mode='std')
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def forward(self, input, mean, std):
        mean_all = torch.mean(input, dim=1, keepdim=True)
        # outputs_mean = self.model_mean(mean.squeeze(2) - mean_all, input - mean_all) + mean_all
        outputs_mean = self.model_mean(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                       self.weight[1]
        outputs_std = self.model_std(std.squeeze(2), input)
        outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
        return outputs


class MLP(nn.Module):
    def __init__(self, input_len, seq_len, pred_len, enc_in, mode):
        super(MLP, self).__init__()
        self.input_len = input_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.mode = mode

        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()

        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.input_len, 512)
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