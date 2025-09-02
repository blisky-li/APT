import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
import numpy as np
from basicts.utils import data_transformation_4_xformer
from typing import Union, List


class iTransformer(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, **model_args):
        super(iTransformer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.output_attention = model_args['output_attention']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.factor = model_args["factor"]
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.embed = model_args['embed']
        self.freq = model_args["freq"]
        self.dropout = model_args["dropout"]
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']


        # self.times_b = nn.Linear(in_features=30, out_features=30, bias=True)
        # self.affine_weight = nn.Parameter(torch.ones(1, 1, model_args['enc_in']))
        # self.affine_bias = nn.Parameter(torch.zeros(1, 1, model_args['enc_in']))
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

        self.save_counter = 0

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                        x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None, dec_self_mask: torch.Tensor = None,
                        dec_enc_mask: torch.Tensor = None, train: bool = True) -> torch.Tensor:
        _, _, N = x_enc.shape
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        # print(enc_out.shape)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]# .permute(0, 2, 1)[:, :, :N]  # filter the covariates

        return dec_out


