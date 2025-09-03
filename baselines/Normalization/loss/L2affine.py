import torch
import torch.nn.functional as F

def l2(x: torch.Tensor, delta: float = 1, lambda_var: float = 1.0) -> torch.Tensor:
    """
    x: 输入张量，形状为 [B, T]（Batch大小, 序列长度）
    delta: 方差阈值（整体变化的最小要求）
    lambda_var: 方差损失的权重
    """
    # 1. 平滑损失（相邻元素的均方差分）
    # 2. 方差损失（防止序列退化为常数）
    # l2 = torch.sum(x ** 2 + 1e-4)
    # 1. 均值归零损失 (按列计算)


    # 展平成 [B, D]
    gamma = 1

    # W_flat = x.view(x.size(0), -1)      # [B, D]
    W_flat = x.reshape(-1)
    # print(W_flat.shape)
    B = W_flat.shape[0]
    # 计算每个维度的无偏方差并加 eps，然后开根号得到 std
    var = W_flat.var(dim=0, unbiased=True)  # [D]
    # print(var)
    std = torch.sqrt(var + 1e-4)              # [D]
    # 对低于 gamma 的部分进行 ReLU 惩罚，并求平均
    # loss = torch.mean(F.relu(gamma - std)) + W_flat.mean().pow(2)
    loss = torch.mean((gamma - std) ** 2) + W_flat.mean().pow(2)# torch.sum(W_flat ** 2) / B
    # print('L2', loss, torch.mean((gamma - std) ** 2), W_flat.mean().pow(2))

    return loss