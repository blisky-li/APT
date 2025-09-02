import torch
from typing import Union, List
import numpy as np




def station_loss(y, statistics_pred, minmax_pred=None):
    bs, len, dim = y.shape
    loss = []
    if minmax_pred is not None:
        min = minmax_pred[:, :, 0].reshape(bs, -1, 1)
        min_max = minmax_pred[:, :, 1].reshape(bs, -1, 1)
        y = (y - min) / min_max
    for k in range(self.args.top_k):
        period_len = self.args.period_list[k]
        if len % period_len != 0:
            length = ((len // period_len) + 1) * period_len
            padding = y[:, -(length - len):, :]
            y = torch.cat([y, padding], dim=1)
        y = y.reshape(bs, -1, period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        stat_true = torch.cat([mean, std], dim=-1)
        loss.append(nn.MSELoss()(statistics_pred[k], stat_true))
        y = y.reshape(bs, -1, dim)[:, :len, :]
    return loss
