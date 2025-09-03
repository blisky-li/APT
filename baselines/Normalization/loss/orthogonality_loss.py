import torch
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def plot_and_save_heatmap(tensor, filename, title="Heatmap", cmap="viridis", dpi=300):
    """
    绘制GPU张量的热力图并保存为图像

    参数:
    tensor (torch.Tensor): GPU上的N×N张量
    filename (str): 保存图像的文件名
    title (str): 图像标题 (默认为"Heatmap")
    cmap (str): 颜色映射 (默认为"viridis")
    dpi (int): 图像分辨率 (默认为300)
    """
    # 确保张量在CPU上并转换为NumPy
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 转换为NumPy数组
    np_array = tensor.detach().numpy()

    # 创建图像
    plt.figure(figsize=(10, 8))

    # 绘制热力图
    plt.imshow(np_array, cmap=cmap)

    # 添加颜色条
    plt.colorbar()

    # 设置标题
    plt.title(title)

    # 添加网格线（可选）
    plt.grid(False)

    # 保存高分辨率图像
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def orthogonality(inputs: List[torch.Tensor]) -> torch.Tensor:
    reg_coef = 1e-4
    total_loss = torch.tensor(0.0).to(inputs[0][0].device)

    for embedding in inputs:
        # embedding = embedding - embedding.mean(dim=0, keepdim=True)
        similarity_matrix = embedding @ embedding.T # embedding @ embedding.T #torch.mm(embedding, embedding.T) / (embedding.size(0) - 1)
        N = similarity_matrix.size(0)

        # 1. 非对角损失：平方和
        mask = ~torch.eye(N, dtype=torch.bool, device=similarity_matrix.device)
        loss_neg = torch.sum(similarity_matrix[mask] ** 2) / (N * (N - 1))  # 移除内部常数

        # 2. 对角损失：强制单位范数
        diag = torch.diag(similarity_matrix)
        loss_pos = torch.sum((1 - diag) ** 2) / N  # 使diag接近1

        # 3. 移除冲突的L2正则项（与loss_pos目标矛盾）
        # L2 = torch.sum(embedding**2)  # 删除此行

        if N != 1:
            loss = loss_neg + loss_pos  # 最终损失
        else:
            loss = loss_pos

        # print(loss_neg, loss_pos)
        total_loss += loss  # 累加损失

    total_loss = total_loss / len(inputs)
    # print(total_loss, 'orth')

    return total_loss


def orthogonality2(inputs: List[torch.Tensor]) -> torch.Tensor:
    reg_coef = 1e-4
    total_loss = torch.tensor(0.0).to(inputs[0][0].device)

    for embedding in inputs:
        N, D = embedding.shape
        # 归一化向量（使点积等于余弦相似度）
        norm_emb = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(norm_emb, norm_emb.T)

        # similarity_matrix = torch.mm(embedding, embedding.T)
        #print(similarity_matrix.max())
        # plot_and_save_heatmap(similarity_matrix, str(N)+'.png')
        # 1. 相邻向量相似性损失（连续性）
        adj_pairs_sim = similarity_matrix.diagonal(-1)  # 相邻向量对的相似度
        adj_pairs_sim2 = similarity_matrix.diagonal(1)
        adjacent_loss = (((0.5 - adj_pairs_sim) ** 2).mean() + ((0.5 - adj_pairs_sim2) ** 2).mean())/2

        # 2. 非相邻向量正交损失（分散性）
        # 创建掩码：只选择非相邻对（|i-j|>1）
        mask = torch.ones((N, N), dtype=torch.bool, device=embedding.device)
        for i in range(N):
            if i > 0: mask[i, i - 1] = False
            if i < N - 1: mask[i, i + 1] = False
            mask[i, i] = False  # 排除自身

        non_adjacent_sim = similarity_matrix[mask]
        orth_loss = torch.sqrt(non_adjacent_sim ** 2 + 1e-4).mean()

        # 3. 自身一致性损失（保持单位长度）
        diag = torch.diag(similarity_matrix)
        self_loss = torch.sqrt((1 - diag) ** 2 + 1e-4).mean()

        # 4. L2正则化（可选）
        l2_reg = torch.sqrt(torch.sum(embedding.std(dim=-1) ** 2) + 1e-4) / N
        # print(adjacent_loss, orth_loss, self_loss, l2_reg)
        # print(adjacent_loss, orth_loss, self_loss)
        total_loss += (adjacent_loss + orth_loss + self_loss)# + l2_reg)
    total_loss = total_loss / len(inputs)

    # print(total_loss)
    # print(total_loss, 'orth')

    return total_loss


def orthogonality3(inputs: List[torch.Tensor]) -> torch.Tensor:
    total_loss = torch.tensor(0.0).to(inputs[0][0].device)

    for embedding in inputs:
        N, D = embedding.shape
        # 1. 归一化向量
        norm_emb = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)

        # 2. 计算余弦相似度矩阵
        sim_matrix = torch.mm(norm_emb, norm_emb.T)  # [N, N]
        # plot_and_save_heatmap(sim_matrix, str(N) + '.png')
        # 3. 创建目标环状相似度矩阵
        indices = torch.arange(N, device=embedding.device)

        # 计算环上距离（考虑首尾相连）
        dist = torch.abs(indices[:, None] - indices[None, :])
        ring_dist = torch.min(dist, N - dist)  # 环上最短距离

        # 创建目标相似度（随距离衰减）
        target_sim = torch.where(
            ring_dist == 0,
            1.0,
            torch.exp(-ring_dist.float()) ) # 衰减系数
        # plot_and_save_heatmap(target_sim, 'target.png')
        # print(target_sim)
        # 4. 计算相似度损失（MSE）
        sim_loss = F.mse_loss(sim_matrix, target_sim)

        # 5. 添加正交惩罚（可选） (正则)
        # 强制非相邻向量正交
        mask = ring_dist > 1
        if mask.sum() > 0:
            orth_loss = (sim_matrix[mask] ** 2).mean()
        else:
            orth_loss = torch.tensor(0.0, device=embedding.device)
        total_loss = sim_loss + orth_loss
    total_loss = total_loss / len(inputs)
    # print(total_loss)
    # print(total_loss, 'orth')
    return total_loss