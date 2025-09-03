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
from ..loss import orthogonality, station_loss, diversity, msn_station_loss, smooth_loss, l2, balance_loss
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)
class NormalizeTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)

        self.normalization_name = cfg['MODEL'].get('NORMALIZE', None)
        self.model_use_tan = cfg['MODEL'].get('USETAN', None)
        self.pred_len = cfg['MODEL']['PARAM']['pred_len']
        self.top_k = cfg['MODEL']['PARAM']['top_k']

        if self.normalization_name == 'SAN':
            self.period_len = cfg['MODEL']['PARAM']['period_len']
        if self.normalization_name == 'SAN' or 'MSN' in self.normalization_name or 'Dish' in self.normalization_name:
            self.station_pretrain_epoch = cfg['MODEL']['PARAM']['station_pretrain_epoch']
            self.station_lambda = cfg['MODEL']['PARAM']['station_lambda']
        if self.model_use_tan:
            self.tan_epoch = cfg['MODEL']['PARAM']['tan_epoch']
            self.station_lambda = cfg['MODEL']['PARAM']['station_lambda']
            self.station_pretrain_epoch = cfg['MODEL']['PARAM']['station_pretrain_epoch']
            self.tan_loss = []
        else:
            self.tan_epoch = 0
        if 'MSN' in self.normalization_name:
            self.period_list = cfg['MODEL']['PARAM']["periods"]

        if self.model_use_tan:
            if self.normalization_name != 'None':
                self.submodules = {
                    'model': self.model.model,  # 注意是self.model的内部成员
                    'normalization': self.model.normalization,
                    'time_embedding': self.model.time_embedding
                }
            else:
                self.submodules = {
                    'model': self.model.model,  # 注意是self.model的内部成员
                    'time_embedding': self.model.time_embedding
                }
        else:
            if self.normalization_name != 'None':
                self.submodules = {
                    'model': self.model.model,  # 注意是self.model的内部成员
                    'normalization': self.model.normalization,
                }
            else:
                self.submodules = {
                    'model': self.model.model,  # 注意是self.model的内部成员
                }


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
        top_list = amps.topk(self.top_k).indices
        period_list = l // top_list
        period_weight = F.softmax(amps[top_list], dim=0)
        self.period_list = period_list
        self.period_weight = period_weight
        # np.save('PEMS04_Top_periods.npy', self.period_list.cpu().detach().numpy())


    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        # TODO: add more preprocessing steps as needed.
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        # rescale data
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        # subset forecasting
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data

    '''def loss_monitor(self, cfg: Dict):'''



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
        # epoch_index = 0
        if self.normalization_name == 'SAN' or 'MSN' in self.normalization_name or 'Dish' in self.normalization_name:
            self.num_epochs += self.station_pretrain_epoch
        if self.model_use_tan:
            self.num_epochs += self.tan_epoch

        epoch_index = self.start_epoch
        early_stop = False
        while epoch_index < self.num_epochs and not early_stop:
            self.tan_loss = []
            # early stopping
            if self.early_stopping_patience is not None and self.current_patience <= 0:
                self.logger.info('Early stopping.')
                early_stop = True
                # break

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

            # MSN get main periods from all training datasets
            '''if epoch == 1: #and 'MSN' in self.normalization_name:
                self._get_period(data_loader)'''

            data_loader = tqdm(data_loader) if get_local_rank() == 0 else data_loader
            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)

            # update lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train/time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

            epoch_index += 1 #$### 非常重要，不能修改！！！！！！！！！！
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

        if self.normalization_name == 'SAN' or self.normalization_name == 'FAN': # For SAN & FAN
            if self.tan_epoch >= epoch  and self.model_use_tan:

                if iter_num % self.iter_per_epoch == 0:
                    print('SAN or MSN Pretraining Mode')

                self._apply_freeze_rules(False, False, True)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                loss = self.station_lambda * (orthogonality(forward_return['orthogonality']) + balance_loss(
                        forward_return['load']) + l2(forward_return['w']) + l2(
                        forward_return['b']))

                self.tan_loss.append(orthogonality(forward_return['orthogonality']).item() + balance_loss(
                        forward_return['load']).item() + l2(forward_return['w']).item() + l2(
                        forward_return['b']).item())

                self.update_epoch_meter('train/loss', loss.item())
                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
                return loss

            else:
                # print('normalization_freeze')
                if iter_num % self.iter_per_epoch == 0:
                    print('SAN OR MSN Training Mode')
                """反向冻结规则"""
                self._apply_freeze_rules(True, True, True)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                # forward_return = self.postprocessing(forward_return)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                loss = self.metric_forward(self.loss, forward_return)
                if self.normalization_name == 'FAN':
                    loss += self.model.normalization.loss(forward_return['target'])
                if self.normalization_name == 'SAN':
                    target = forward_return['target'].squeeze()
                    if isinstance(self.model.normalization, torch.nn.ModuleList):
                        loss += station_loss(target, self.model.normalization[0].station_pred, self.period_len)
                    else:
                        loss += station_loss(target, self.model.normalization.station_pred, self.period_len)

                self.update_epoch_meter('train/loss', loss.item())
                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
                return loss


        elif self.model_use_tan:  # DishTS & RevIN:
            if epoch <= self.tan_epoch:

                if iter_num % self.iter_per_epoch == 0:
                    print('Regular APT pretraining Mode')

                self._apply_freeze_rules(False, False, True)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if iter_num % self.iter_per_epoch == 0:
                    print('None MODE,TAN')

                loss = self.station_lambda * (orthogonality(forward_return['orthogonality']) + balance_loss(
                        forward_return['load']) + l2(forward_return['w']) + l2(
                        forward_return['b']))

                self.update_epoch_meter('train/loss', loss.item())

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

            elif epoch == (self.tan_epoch + 1):

                if iter_num % self.iter_per_epoch == 0:
                    print('Regular APT pretraining Mode')

                self._apply_freeze_rules(False, False, True)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if iter_num % self.iter_per_epoch == 0:
                        print('None MODE,TAN')

                loss = self.metric_forward(self.loss, forward_return)

                self.update_epoch_meter('train/loss', loss.item())

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)

                    self.update_epoch_meter(f'train/{metric_name}', metric_item.item())


            else:
                if iter_num % self.iter_per_epoch == 0:
                    print('Regular APT Training Mode')

                self._apply_freeze_rules(True, True, False)

                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

                if iter_num % self.iter_per_epoch == 0:
                    print('None MODE ORI')

                loss = self.metric_forward(self.loss, forward_return)
                self.update_epoch_meter('train/loss', loss.item())

                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, forward_return)
                    self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

            return loss

        else:
            if iter_num % self.iter_per_epoch == 0:
                print('Regular Training Mode')
            self._apply_freeze_rules(True, True, False)

            forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
            # forward_return = self.postprocessing(forward_return)
            if self.cl_param:
                cl_length = self.curriculum_learning(epoch=epoch)
                forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

            if self.model_use_tan:
                loss = self.metric_forward(self.loss, forward_return)
            else:
                loss = self.metric_forward(self.loss, forward_return)
            if self.normalization_name == 'FAN':
                loss += self.model.normalization.loss(forward_return['target'])
            self.update_epoch_meter('train/loss', loss.item())
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, forward_return)
                self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

            return loss

    @master_only
    def save_model(self, epoch: int):
        """Save checkpoint for all submodules and meta information."""

        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        # 保存每个模块的 state_dict
        for name, module in self.submodules.items():
            ckpt_path = self.get_ckpt_path(epoch, name)
            torch.save(module.state_dict(), ckpt_path)

        # 保存 meta 信息
        meta = {
            'epoch': epoch,
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics
        }
        meta_path = os.path.join(self.ckpt_save_dir,
                                 f"{self.model_name}_meta_{str(epoch).zfill(len(str(self.num_epochs)))}.pt")
        torch.save(meta, meta_path)

        # ---------- 修改后的备份逻辑 ----------
        last_epoch = epoch - 1
        for name in self.submodules.keys():
            last_ckpt_path = self.get_ckpt_path(last_epoch, name)
            backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # meta 也备份
        last_meta_path = os.path.join(self.ckpt_save_dir,
                                      f"{self.model_name}_meta_{str(last_epoch).zfill(len(str(self.num_epochs)))}.pt")
        backup_last_ckpt(last_meta_path, epoch, self.ckpt_save_strategy)

        # ---------- 清理逻辑 ----------
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)  # 建议你清理时匹配 {model_name}_*_{epoch}.pt 的格式

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)

        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            ckpt_dir = self.ckpt_save_dir

            for name, module in self.submodules.items():
                ckpt_path = os.path.join(ckpt_dir, f"{self.model_name}_{name}_best_{metric_name.replace('/', '_')}.pt")
                torch.save(module.state_dict(), ckpt_path)

            meta = {
                'epoch': epoch,
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            meta_path = os.path.join(ckpt_dir, f"{self.model_name}_meta_best_{metric_name.replace('/', '_')}.pt")
            torch.save(meta, meta_path)

            self.current_patience = self.early_stopping_patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load state dicts for all submodules.

        Args:
            ckpt_path (str, optional): any one of the module ckpt paths (e.g., model_best_val_MAE.pt),
                or a manually constructed single string indicating the checkpoint set to load.
        """
        try:
            if ckpt_path is not None:
                base = os.path.basename(ckpt_path)

                # 是否为 best 模型
                if '_best_' in base:
                    suffix = base.split('_best_')[-1].replace('.pt', '')  # e.g., val_MAE
                    for name, module in self.submodules.items():
                        sub_path = os.path.join(
                            self.ckpt_save_dir,
                            f"{self.model_name}_{name}_best_{suffix}.pt"
                        )
                        module.load_state_dict(torch.load(sub_path), strict=strict)

                # 是否为 epoch 模型
                elif base.endswith('.pt') and base.split('_')[-1].replace('.pt', '').isdigit():
                    epoch = int(base.split('_')[-1].replace('.pt', ''))
                    for name, module in self.submodules.items():
                        sub_path = self.get_ckpt_path(epoch, name)
                        module.load_state_dict(torch.load(sub_path), strict=strict)
                else:
                    raise ValueError(f"Unsupported ckpt_path format: {ckpt_path}")

            else:
                # 自动找出最近 epoch
                epoch = self._get_latest_epoch()
                for name, module in self.submodules.items():
                    sub_path = self.get_ckpt_path(epoch, name)
                    module.load_state_dict(torch.load(sub_path), strict=strict)

        except (OSError, KeyError, ValueError) as e:
            raise OSError(f"Failed to load model from checkpoint: {e}") from e

    def load_model_resume(self, strict: bool = True):
        """Resume from last checkpoint if available."""

        try:
            epoch = self._get_latest_epoch()

            # 加载模型每个子模块
            for name, module in self.submodules.items():
                path = self.get_ckpt_path(epoch, name)
                module.load_state_dict(torch.load(path), strict=strict)

            # 加载优化器和meta信息
            meta_path = os.path.join(
                self.ckpt_save_dir,
                f"{self.model_name}_meta_{str(epoch).zfill(len(str(self.num_epochs)))}.pt"
            )
            meta = torch.load(meta_path)
            self.optim.load_state_dict(meta['optim_state_dict'])
            self.start_epoch = meta['epoch']
            self.best_metrics = meta.get('best_metrics', {})

            if self.scheduler is not None:
                self.scheduler.last_epoch = meta['epoch']
            self.logger.info(f'Resume training from epoch {epoch}')

        except (FileNotFoundError, IndexError, OSError, KeyError) as e:
            self.logger.warning(f"No checkpoint found, start training from scratch. ({e})")
            self.start_epoch = 0
            self.best_metrics = {}

    def _get_latest_epoch(self) -> int:
        """Scan ckpt_save_dir for latest epoch."""
        files = os.listdir(self.ckpt_save_dir)
        epochs = []
        for f in files:
            if f.startswith(f"{self.model_name}_meta_") and f.endswith('.pt'):
                try:
                    ep = int(f.replace(f"{self.model_name}_meta_", '').replace('.pt', ''))
                    epochs.append(ep)
                except ValueError:
                    continue
        if not epochs:
            raise FileNotFoundError("No checkpoint meta files found.")
        return max(epochs)

    def _get_epoch_from_ckpt_path(self, ckpt_path: str) -> int:
        """Derive epoch from custom ckpt_path, or use latest."""
        if ckpt_path is not None:
            base = os.path.basename(ckpt_path)
            try:
                return int(base.replace('.pth', '').split('_')[-1])
            except Exception as e:
                raise ValueError(f"Cannot parse epoch from ckpt_path: {ckpt_path}") from e
        else:
            return self._get_latest_epoch()

    def get_ckpt_path(self, epoch: int, module_name: str) -> str:
        """Get checkpoint path for each module.

        Format: "{ckpt_save_dir}/{model_name}_{module_name}_{epoch}.pt"
        """
        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        ckpt_name = f"{self.model_name}_{module_name}_{epoch_str}.pt"
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        valid_epoch = -1
        if self.model_use_tan:
            valid_epoch += self.tan_epoch

        if train_epoch is not None and train_epoch >= valid_epoch:
            self.save_best_model(train_epoch, 'val/' + self.target_metrics, greater_best=greater_best)

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
        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)  # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return['outputs']}
        else:
            model_returns = dict()
            model_returns['prediction'] = model_return['outputs']
            embedding_lst = model_return['time_embedding']
            if self.model_use_tan:
                model_returns['orthogonality'] = embedding_lst
                model_returns['stat_pred'] = model_return['stat_pred']
                model_returns['w'] = model_return['w']
                model_returns['b'] = model_return['b']
                model_returns['load'] = model_return['load']
            model_return = model_returns
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."
        # print(model_return.keys())
        model_return = self.postprocessing(model_return)
        # a = orthogonality(model_return['orthogonality'])
        return model_return
