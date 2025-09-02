import torch
import torch.nn as nn

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack


class Informer(nn.Module):
    """
    Paper: Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting
    Link: https://arxiv.org/abs/2012.07436
    Ref Official Code: https://github.com/zhouhaoyi/Informer2020
    """

    def __init__(self,  **model_args):
        super(Informer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.label_len = model_args['label_len']
        self.attn = model_args['attn']
        self.output_attention = model_args['output_attention']

        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.d_model = model_args['d_model']
        self.d_ff = model_args['d_ff']
        self.mix = model_args['mix']
        self.dropout = model_args['dropout']
        self.factor = model_args['factor']

        self.activation = model_args['activation']
        self.n_heads = model_args['n_heads']
        self.distil = model_args['distil']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']

        self.num_time_features = model_args['num_time_features']
        self.time_of_day_size =model_args['time_of_day_size']
        self.day_of_week_size = model_args['day_of_week_size']
        self.day_of_month_size = model_args['day_of_month_size']
        self.day_of_year_size = model_args['day_of_year_size']
        self.embed = model_args['embed']

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.time_of_day_size, self.day_of_week_size, self.day_of_month_size, self.day_of_year_size,
                                           self.embed, self.num_time_features, self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.time_of_day_size, self.day_of_week_size, self.day_of_month_size, self.day_of_year_size,
                                           self.embed, self.num_time_features, self.dropout)
        # Attention
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention),
                                self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers-1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                                self.d_model, self.n_heads, mix=self.mix),
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                                self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None, train: bool = True) -> torch.Tensor:
        """Feed forward of Informer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in Informer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :]  # [B, L, N, C]



class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True, num_time_features=-1):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.label_len = int(label_len)
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, num_time_features, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, num_time_features, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:
        """Feed forward of Informer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in Informer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, N, C]


