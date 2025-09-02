import torch
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def balance_loss(load):
    """计算负载均衡损失（参与梯度传播）"""
    # 合并两个时间点的权重
    total_weights = load + 1e-8
    # print(load.shape)
    # 计算当前批次的原型概率分布
    p = total_weights / total_weights.sum()
    # print(p)

    # 目标均匀分布
    num_proto = load.shape[0]
    uniform = torch.ones_like(p) / num_proto
    # print(p)
    # KL散度损失 (p -> uniform)
    # loss = F.kl_div(p.log(), uniform, reduction='batchmean')
    loss = torch.sum((p - uniform)**2)

    # 添加可调节系数
    return loss