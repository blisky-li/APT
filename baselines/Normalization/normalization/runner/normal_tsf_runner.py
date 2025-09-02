import os
import json
import math
import time
import inspect
import functools
import torch
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Union, Optional, Dict
from tqdm import tqdm
from basicts.runners import BaseTimeSeriesForecastingRunner
from easytorch.utils import TimePredictor, get_local_rank, is_master, master_only
from ..loss import orthogonality, station_loss, diversity

class NormalizeTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.normalization_name = cfg['MODEL'].get('NORMALIZE', None)
        self.model_use_tan = cfg['MODEL'].get('USETAN', None)
        self.pred_len = cfg['MODEL']['PARAM']['pred_len']

        if self.normalization_name == 'SAN':
            self.period_len = cfg['MODEL']['PARAM']['period_len']
        if self.normalization_name == 'SAN' or 'MSN' in self.normalization_name:
            self.station_pretrain_epoch = cfg['MODEL']['PARAM']['station_pretrain_epoch']
            self.station_lambda = cfg['MODEL']['PARAM']['station_lambda']
        if self.model_use_tan:
            self.tan_epoch = cfg['MODEL']['PARAM']['tan_epoch']
            self.station_lambda = cfg['MODEL']['PARAM']['station_lambda']
            self.station_pretrain_epoch = cfg['MODEL']['PARAM']['station_pretrain_epoch']
        else:
            self.tan_epoch = 0

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C].

        Returns:
            torch.Tensor: Data with selected features.
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with arbitrary shape.

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C].
        """

        data = data[:, :, :, self.target_features]
        return data

    def _get_period(self, train_loader):
        """
        get top k periodicity
        """

        amps = 0.0
        count = 0
        for data in train_loader:

            lookback_window = data['inputs'][:, :, :, 0]
            # print(lookback_window.shape)
            b, l, dim = lookback_window.size()
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
            count += 1
        amps = amps / count
        amps[0] = 0
        max_period = self.pred_len * 2
        max_freq = l // max_period + 1
        amps[0:max_freq] = 0
        top_list = amps.topk(20).indices
        period_list = l // top_list
        period_weight = F.softmax(amps[top_list], dim=0)
        self.period_list = period_list
        self.period_weight = period_weight
        # print(self.period_list)
        # print(self.period_weight)

    def train(self, cfg: Dict):
        """Train model.

        Train process:
        [init_training]
        for in train_epoch
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]

        Args:
            cfg (Dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        # training loop
        epoch_index = 0
        if self.normalization_name == 'SAN' or 'MSN' in self.normalization_name:
            self.num_epochs += self.station_pretrain_epoch
        if self.model_use_tan:
            self.num_epochs += self.tan_epoch
        for epoch_index in range(self.start_epoch, self.num_epochs):
            # early stopping
            if self.early_stopping_patience is not None and self.current_patience <= 0:
                self.logger.info('Early stopping.')
                break

            epoch = epoch_index + 1
            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            if cfg.get('TRAIN.DATA.DEVICE_PREFETCH', False):
                data_loader = DevicePrefetcher(self.train_data_loader)
            else:
                data_loader = self.train_data_loader

            if epoch == 1:
                self._get_period(data_loader)

            data_loader = tqdm(data_loader) if get_local_rank() == 0 else data_loader



            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train_time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end(cfg=cfg, train_epoch=epoch_index + 1)

    def _apply_freeze_rules(self,  model, normalization, tan):
        """统一参数冻结逻辑"""

        rules = {
            'model.': model,  # 解冻模型主体
            'normalization.': normalization,  # 解冻归一化
            'time_embedding.': tan  # TAN冻结
        }

        # 应用规则
        for name, param in self.model.named_parameters():
            for prefix, grad_flag in rules.items():
                if name.startswith(prefix):
                    param.requires_grad = grad_flag
                    break

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        data = self.preprocessing(data)



        if self.normalization_name == 'SAN' or 'MSN' in self.normalization_name:
            if self.tan_epoch < epoch <= self.station_pretrain_epoch + self.tan_epoch and self.model_use_tan:
                self._apply_freeze_rules(False, False, True)
                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                forward_return = self.postprocessing(forward_return)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                '''if 'lambda' not in forward_return.keys() and self.model_use_tan:
                    loss = self.metric_forward(self.loss, forward_return) + self.station_lambda * orthogonality(
                        forward_return['orthogonality'])
                else:'''
                if self.model_use_tan:
                    loss = (self.metric_forward(self.loss, forward_return) + self.station_lambda * orthogonality(forward_return['orthogonality'])) * 10# self.metric_forward(self.loss, forward_return)
                else:
                    loss = self.metric_forward(self.loss, forward_return)

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train_{metric_name}', metric_item.item())
                return loss


            elif epoch <= self.station_pretrain_epoch:
                # print('model_freeze')
                """统一冻结逻辑"""
                # 定义冻结规则：冻结模型主体和时间嵌入，解冻归一化模块
                self._apply_freeze_rules(False, True, False)

                # print(prediction.shape)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                forward_return = self.postprocessing(forward_return)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                target = forward_return['target'].squeeze()

                model_station_loss = station_loss(target, self.model.normalization.station_pred, self.period_len)
                loss = model_station_loss * self.station_lambda
                '''if self.model_use_tan:
                    loss = loss + self.station_lambda * orthogonality(forward_return['orthogonality'])'''
                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train_{metric_name}', metric_item.item())
                # print(loss)
                return loss

            else:
                # print('normalization_freeze')
                """反向冻结规则"""
                self._apply_freeze_rules(True, False, True)

                '''
                                elif self.station_pretrain_epoch + 1 < epoch <= self.station_pretrain_epoch + 5:
                                    for name, param in self.model.named_parameters():
                                        if name.startswith('model.'):  # 根据前缀识别模块
                                            param.requires_grad = True
                                    if hasattr(self.model, 'normalization'):
                                        for param in self.model.normalization.parameters():
                                            param.requires_grad = False
                                    # 冻结时间嵌入模块
                                    if self.model.use_TAN:
                                        for param in self.model.time_embedding.parameters():
                                            param.requires_grad = True
                                '''
                # print(prediction.shape)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                forward_return = self.postprocessing(forward_return)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if self.model_use_tan:
                    loss = self.metric_forward(self.loss, forward_return) + self.station_lambda * orthogonality(
                        forward_return['orthogonality'])
                else:
                    loss = self.metric_forward(self.loss, forward_return)

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train_{metric_name}', metric_item.item())
                return loss

        elif self.model_use_tan and self.tan_epoch != 0:
            if epoch <= self.tan_epoch:
                self._apply_freeze_rules(False, True, True)
                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                forward_return = self.postprocessing(forward_return)
                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if 'lambda' in forward_return.keys() and self.model_use_tan:
                    loss = self.metric_forward(self.loss,forward_return)#  + self.station_lambda * orthogonality(forward_return['orthogonality'])
                else:
                    loss = self.metric_forward(self.loss, forward_return)

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train_{metric_name}', metric_item.item())
            else:

                # self._apply_freeze_rules(True, True, False)
                self._apply_freeze_rules(True, True, True)
                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                forward_return = self.postprocessing(forward_return)
                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if 'lambda' in forward_return.keys() and self.model_use_tan:
                    loss = self.metric_forward(self.loss, forward_return) + self.station_lambda * (orthogonality(forward_return['orthogonality']) + diversity(forward_return['orthogonality']))#+ self.station_lambda * diversity(forward_return['orthogonality'])# self.station_lambda * (orthogonality(forward_return['orthogonality']) + diversity(forward_return['orthogonality']))
                else:
                    loss = self.metric_forward(self.loss, forward_return)

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train_{metric_name}', metric_item.item())

            return loss


        else:
            self._apply_freeze_rules(True, True, False)
            forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
            forward_return = self.postprocessing(forward_return)
            if self.cl_param:
                cl_length = self.curriculum_learning(epoch=epoch)
                forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

            if 'lambda' in forward_return.keys() and self.model_use_tan:
                loss = self.metric_forward(self.loss, forward_return)# + self.station_lambda * 5 * orthogonality(forward_return['orthogonality'])
            else:
                loss = self.metric_forward(self.loss, forward_return)

            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, forward_return)
                self.update_epoch_meter(f'train_{metric_name}', metric_item.item())

            return loss



    def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, freeze_mode=None, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)  # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        '''if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])'''

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)




        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        else:
            model_returns = dict()
            model_returns['prediction'] = model_return[0]
            '''if model_return[1].ndimension() == 0:
                model_returns['lambda'] = model_return[1]
            else:
                model_returns['lambda'] = torch.tensor(0.001).to(model_returns['prediction'].device)'''
            embedding_lst = []
            for i in range(2, len(model_return)):
                embedding = model_return[i]
                # embedding = embedding / embedding.norm(dim=1, keepdim=True)
                similarity_embedding = torch.mm(embedding, embedding.T)
                N = similarity_embedding.size(0)
                # print(N, similarity_embedding.device, similarity_embedding.shape)
                identity_matrix = torch.eye(N).to(similarity_embedding.device)
                # print(similarity_embedding)
                embedding_lst.append([similarity_embedding, identity_matrix])
            model_returns['orthogonality'] = embedding_lst
            model_return = model_returns
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)


        #print(self.normalization_name)
        '''if self.normalization_name == 'SAN':
            print(self.model.normalization.station_pred.shape)'''


        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."
        # print(model_return.keys())
        return model_return
