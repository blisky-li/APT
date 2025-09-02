import torch
from typing import Union, List
import numpy as np

def orthogonality(inputs: List[torch.Tensor]) -> torch.Tensor:

    total_loss = torch.tensor(0.0).to(inputs[0][0].device)

    for similarity_matrix, identity_matrix in inputs:
        # print(similarity_matrix.shape)
        # 计算 Frobenius 范数的损失
        # loss = torch.norm(similarity_matrix - identity_matrix, p='fro') ** 2 / (similarity_matrix.numel())  # Frobenius范数
        loss = torch.mean(((similarity_matrix - identity_matrix) ** 2))/2 + torch.mean(similarity_matrix ** 2)/2#  / similarity_matrix.numel()
        total_loss += loss  # 累加损失
    # print(total_loss)

    return total_loss