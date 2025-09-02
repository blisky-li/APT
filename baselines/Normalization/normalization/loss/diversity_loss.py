import torch
from typing import Union, List
import numpy as np

def diversity(inputs: List[torch.Tensor]) -> torch.Tensor:

    total_loss = torch.tensor(0.0).to(inputs[0][0].device)

    for similarity_matrix, identity_matrix in inputs:
        # print(similarity_matrix.shape)
        # 计算 Frobenius 范数的损失
        # loss = torch.norm(similarity_matrix - identity_matrix, p='fro') ** 2 / (similarity_matrix.numel())  # Frobenius范数
        loss = torch.mean(torch.triu(similarity_matrix, diagonal=1)**2)
        total_loss += loss  # 累加损失
    # print(total_loss)

    return total_loss