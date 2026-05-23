import torch
import numpy as np
from pathlib import Path
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from torch import nn, optim
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import wandb
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

def model_sparsity(model, threshold):
    cnt = 0
    tot_cnt = 0
    for p in model.parameters():
        cnt += (p.data > threshold).sum().item() + (p.data < -threshold).sum().item()
        tot_cnt += torch.ones(p.size()).sum().item()
    return 1. - float(cnt) / float(tot_cnt)

class ELLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, order=1, kappa=0.0, lambd=0.0, exp_bias=0.0, l1_kappa=0.0, momentum=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8, theta_grad=False, theta_in_buffer=False,
                 base_optimizer_type='SGD', update_kappa=False, ell_eps=0.0, ell_m=-1):
        self.lr = lr
        self.order = order
        self.ell_m = ell_m if ell_m >= 0 else (order - 1.)
        print(order, 'ell_m=', self.ell_m)
        self.kappa = kappa
        self.l1_kappa = l1_kappa
        self.lambd = lambd
        self.exp_bias = exp_bias
        self.exp_order = 2. * (order - 1) / order
        self.momentum = momentum
        self.ell_eps = ell_eps
        self.state = {}
        # self.momentum_buf = {}
        # self.exp_avgs = {}
        # self.exp_avg_sqs = {}
        # self.denom_buffer = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.current_step = 0
        self.eps = eps
        self.theta_grad = theta_grad
        self.theta_in_buffer = theta_in_buffer
        self.retrain = False
        self.base_optimizer_type = base_optimizer_type
        self.update_kappa = update_kappa
        super().__init__(params, {"lr": lr})

        for param_group in self.param_groups:
            for p in param_group['params']:
                self.state[p] = {}

    def get_lr(self):
        return self.lr

    def schedule_kappa(self, new_kappa):
        self.kappa = new_kappa

    def schedule(self, new_lr):
        w = new_lr / self.lr
        self.lr = self.lr * w
        if self.update_kappa:
            self.kappa = self.kappa * w
            self.l1_kappa = self.l1_kappa * w

    def step(self, closure=False):
        self.current_step += 1
        for param_group in self.param_groups:
            params = param_group['params']
            group_lr = param_group['lr']
            group_kappa = param_group['kappa']
            group_l1_kappa = param_group['l1_kappa']
            momentum_buf = []
            exp_avgs = []
            exp_avg_sqs = []
            for i, p in enumerate(params):
                state = self.state[p]
                if 'momentum_buffer' in state:
                    momentum_buf.append(state['momentum_buffer'])
                if 'exp_avg_buffer' in state:
                    exp_avgs.append(state['exp_avg_buffer'])
                if 'exg_avg_sq_buffer' in state:
                    exp_avg_sqs.append(state['exp_avg_sq_buffer'])

            # self.denom_buffer = []


            for i, p in enumerate(params):
                d_p = torch.zeros(p.size())
                p_1_k = torch.pow(torch.abs(p.data), 1. / self.order) * torch.sign(p.data)
                if p.grad is not None:
                    uk = self.order * torch.pow(torch.abs(p_1_k), self.order - 1.)
                    if self.ell_eps <= 0.0:
                        scale = torch.pow(p_1_k, self.ell_m)
                    else:
                        scale = uk / (uk + self.ell_eps) * torch.pow(p_1_k, self.ell_m)
                    d_p = p.grad * scale

                alpha_ = torch.ones(p.size()).to(p.device)
                if self.theta_grad:
                    if self.theta_in_buffer:
                        d_p = d_p * torch.pow(torch.abs(p_1_k), self.order - 1.0) * self.order * torch.sign(p_1_k)
                    else:
                        alpha_ = alpha_ * torch.pow(torch.abs(p_1_k), self.order - 1.0) * self.order * torch.sign(p_1_k)
                # print(d_p)
                # print(p_1_k)
                if 'mask' in self.state[p]:
                    p_mask = self.state[p]['mask']
                    # print('MASK IS SELECTED!')
                else:
                    p_mask = torch.ones_like(p_1_k).to(p_1_k.device)
                    
                if self.base_optimizer_type == 'SGD':
                    d_p += p_1_k * group_kappa * p_mask + torch.sign(p_1_k) * group_l1_kappa * p_mask
                    res = p_1_k
                else:
                    res = p_1_k - p_1_k * group_kappa - torch.sign(p_1_k) * group_l1_kappa

                # SGD
                tmp = torch.zeros(p.size())
                if len(momentum_buf) > i:
                    tmp = momentum_buf[i]

                tmp = tmp.to(d_p.device)
                tmp = tmp * self.momentum + d_p
                # tmp_buf.append(tmp)
                self.state[p]['momentum_buffer'] = tmp

                # AdamW
                exp_avg = exp_avgs[i] if len(exp_avgs) > i else torch.zeros_like(p)
                exp_avg_sq = exp_avg_sqs[i] if len(exp_avg_sqs) > i else torch.zeros_like(p)
                exp_avg = exp_avg.to(d_p.device)
                exp_avg_sq = exp_avg_sq.to(d_p.device)
                exp_avg.mul_(self.beta1).add_(d_p, alpha=1 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(d_p, d_p, value=1 - self.beta2)

                # tmp_exp_avgs.append(exp_avg)
                # tmp_exp_avg_sqs.append(exp_avg_sq)
                self.state[p]['exp_avg_buffer'] = exp_avg
                self.state[p]['exp_avg_sq_buffer'] = exp_avg_sq
                bias_correction1 = 1 - self.beta1 ** self.current_step
                bias_correction2 = 1 - self.beta2 ** self.current_step
                step_size = group_lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                # self.denom_buffer.append(denom - self.eps)
                if self.base_optimizer_type == 'SGD':
                    res.add_(tmp * alpha_, alpha=-group_lr)
                else:
                    res.addcdiv_(exp_avg * alpha_, denom, value=-step_size)

                # print(p)
                # print(torch.norm(p_1_k) ** 2)
                # print(p.shape)
                # print(p_1_k.shape)
                # p.data = torch.pow(p_1_k - p_1_k * self.kappa - torch.sign(p_1_k) * self.l1_kappa - self.lr * tmp, self.order)
                if self.retrain:
                    p.data = torch.pow(res * self.retrain_mask[i], self.order)
                else:
                    p.data = torch.pow(torch.abs(res), self.order) * torch.sign(res)

    def sparsity(self, threshold):
        cnt = 0
        tot_cnt = 0
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                cnt += (p.data > threshold).sum().item() + (p.data < -threshold).sum().item()
                tot_cnt += torch.ones(p.size()).sum().item()
        return 1. - float(cnt) / float(tot_cnt)

    def print_weight(self):
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                print(p)
                print(p.data)