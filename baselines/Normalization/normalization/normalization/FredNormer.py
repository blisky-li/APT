import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

class FredNormer(nn.Module):
    def __init__(self, **model_args):
        super(FredNormer, self).__init__()
        self.num_channels = model_args['enc_in']
        self.seq_length = model_args['seq_len']
        self.freq_length = model_args['seq_len'] // 2 + 1

        # 定义可学习的权重和偏置
        self.W_r = nn.Parameter(torch.randn(self.freq_length, self.num_channels))
        self.B_r = nn.Parameter(torch.zeros(self.freq_length, self.num_channels))
        self.W_i = nn.Parameter(torch.randn(self.freq_length, self.num_channels))
        self.B_i = nn.Parameter(torch.zeros(self.freq_length, self.num_channels))

    def compute_stability(self, x):
        # 计算频率稳定性度量
        fft_x = fft.rfft(x, dim=1)
        amplitude = torch.abs(fft_x)

        mean = torch.mean(amplitude, dim=0)
        std = torch.std(amplitude, dim=0)

        stability = mean / (std + 1e-5)  # 添加小值以避免除零
        return stability

    def forward(self, x):
        # 应用一阶差分
        x_diff = torch.diff(x, dim=1, prepend=x[:, :1])
        # 计算FFT
        fft_x = fft.rfft(x_diff, dim=1)
        # 计算稳定性度量
        stability = self.compute_stability(x)
        # 分离实部和虚部
        real = fft_x.real
        imag = fft_x.imag

        # 应用频率稳定性加权
        real = real * (stability * self.W_r + self.B_r)
        imag = imag * (stability * self.W_i + self.B_i)

        # 重构复数FFT
        fft_weighted = torch.complex(real, imag)

        # 应用逆FFT
        x_normalized = fft.irfft(fft_weighted, n=self.seq_length, dim=1)

        return x_normalized
