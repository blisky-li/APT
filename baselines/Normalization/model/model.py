import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicts.utils import data_transformation_4_xformer
from typing import Union, List

from ..arch import iTransformer, Informer, Autoformer, DLinear, PatchTST, UMixer
from ..normalization import DAIN, RevIN, DishTS, APT, FAN, SAN

model_dict = {
    'Informer': Informer,
    'iTransformer': iTransformer,
    'Autoformer': Autoformer,
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'UMixer': UMixer,
}

normalization_dict = {
    'DAIN': DAIN,
    'RevIN': RevIN,
    'DishTS': DishTS,
    'FAN': FAN,
    'SAN': SAN,
}

class ModelWithNormalization(nn.Module):
    def __init__(self, **model_args):
        super(ModelWithNormalization, self).__init__()
        self.model_name = model_args['model_name']
        print(self.model_name, model_dict[self.model_name])
        self.model = model_dict[self.model_name](**model_args)
        self.normalization_name = model_args['normalization_name']
        if self.normalization_name in normalization_dict.keys():
            self.normalization = normalization_dict[self.normalization_name](**model_args)
        self.use_TAN = model_args['use_tan']
        if self.use_TAN:
            self.time_embedding = APT(**model_args)
            self.station_lambda = model_args['station_lambda']


        self.is_xformer = model_args['is_xformer']
        self.use_TAN = model_args['use_tan']

        self.label_len = model_args['label_len']




    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                                 future_data=future_data,
                                                                                 start_token_len=int(self.label_len))

        # x_enc.shape, B,L,N
        if self.normalization_name in normalization_dict.keys():
            x_enc = self.normalization(x_enc, 'norm')
        # print(x_enc.shape, x_dec.shape, x_mark_enc.shape)
        if self.use_TAN:
            times_w, times_b = self.time_embedding(x_mark_enc, x_mark_dec)

            x_enc = x_enc * times_w + times_b
        if self.is_xformer:

            dec_out = self.model(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec, train=train)
        else:
            dec_out = self.model(x_enc, future_data, batch_seen, epoch, train, **kwargs)
        # print(dec_out.shape)
        if self.use_TAN:
            # print(times_w.shape)
            # print(dec_out.shape, times_w.shape)
            dec_out = (dec_out - times_b) / times_w
        if self.normalization_name in normalization_dict.keys():
            dec_out = self.normalization(dec_out, 'denorm')

        # dec_out.shape, B,L,N
        if not self.use_TAN:
            return dec_out.unsqueeze(-1)
        else:
            return [
                dec_out.unsqueeze(-1),
                self.station_lambda,
                self.time_embedding.get_combined_embeddings()]