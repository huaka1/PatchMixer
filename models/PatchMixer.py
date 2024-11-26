__all__ = ['PatchMixer']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from torch.distributions.normal import Normal
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP


class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        # dim=self.patch_num, a=self.a, kernel_size=self.kernel_size

        # 深度可分离卷积块 (Depthwise Separable Convolution)
        self.Resnet = nn.Sequential(
            # group=dim，表示每个通道独立进行卷积操作
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        
        # 点卷积块 (Pointwise Convolution)
        self.Conv_1x1 = nn.Sequential(
            # 1x1卷积层：用于改变通道数，实现跨通道信息交互
            # 将通道数从dim降至a，相当于降维操作
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                    # x: [batch * n_val, a, d_model]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x, loss_coef=1e-2):
        x, balance_loss = self.model(x, loss_coef)
        return x, balance_loss


class Backbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        # if configs.a < 1 or configs.a > self.patch_num:
        #     configs.a = self.patch_num
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers

        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))

        self.W_P = nn.Linear(self.patch_size, self.d_model)
        self.head0 = nn.Sequential(
            # 将输入张量从倒数第二个维度开始展平
            # 例如输入形状为[batch * n_val, patch_num, d_model]
            # 输出形状为[batch * n_val, patch_num * d_model]
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
        
        # 添加路由相关组件
        self.num_experts = 4  # 可以根据需要调整专家数量
        self.k = 2           # 选择的专家数量   
        self.noisy_gating = True
        
        # 添加路由所需的层
        self.start_linear = nn.Linear(in_features=self.nvals, out_features=1)
        self.w_noise = nn.Linear(self.lookback, self.num_experts)
        self.w_gate = nn.Linear(self.lookback, self.num_experts)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0] # (128, 96, 7)
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2): # [B, L, N]
        x = self.start_linear(x).squeeze(-1) # [B, L, N] -> [B, L]

        # clean_logits = x @ self.w_gate
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        # x = x.permute(0, 2, 1)  # x: [batch, n_val, seq_len]

        # 添加路由操作
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        
        # 计算 balance loss
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        
        # 专家模型处理
        expert_outputs = []
        for i in range(self.num_experts):
            expert_input = expert_inputs[i]
            # 对每个专家输入执行原始的处理流程
            B, _, _ = expert_input.shape
            expert_input = expert_input.permute(0, 2, 1)
            x_lookback = self.padding_patch_layer(expert_input)
            x_expert = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x_expert = self.W_P(x_expert)
            x_expert = torch.reshape(x_expert, (-1, x_expert.shape[2], x_expert.shape[3]))
            x_expert = self.dropout(x_expert)
            
            u = self.head0(x_expert)
            
            for PatchMixer_block in self.PatchMixer_blocks:
                x_expert = PatchMixer_block(x_expert)
            x_expert = self.head1(x_expert)
            x_expert = u + x_expert

            x_expert = torch.reshape(x_expert, (B, nvars, -1))
            x_expert = x_expert.permute(0, 2, 1) 
            
            expert_outputs.append(x_expert)

        # 合并专家输出
        x = dispatcher.combine(expert_outputs) # 104, 96, 7, 4
        
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x, balance_loss
    