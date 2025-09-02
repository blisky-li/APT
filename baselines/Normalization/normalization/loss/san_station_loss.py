import torch
from typing import Union, List
import numpy as np
import torch.nn as nn

def station_loss(y, statistics_pred, period_len) -> torch.Tensor:

    bs, len, dim = y.shape
    y = y.reshape(bs, -1,  period_len, dim)
    mean = torch.mean(y, dim=2)
    std = torch.std(y, dim=2)
    station_ture = torch.cat([mean, std], dim=-1)
    loss = nn.MSELoss()(statistics_pred, station_ture)
    return loss
