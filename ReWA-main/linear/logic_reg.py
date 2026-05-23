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
from dataset import GaussianDataset, LogicRegDataset



def train_wrapper(config, args):
    args.__dict__.update(config)
    train(args)

def train(args):
    seed = args.seed
    test_seed = args.test_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    threshold_list = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 1e-3]
    training_data = LogicRegDataset(seed=seed, num_samples=args.dataset_len, dim=args.dim)
    test_data = LogicRegDataset(seed=test_seed, num_samples=args.dataset_len, dim=args.dim)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model = Network(args)
    model.to(device)
    # print(args.theta_in_buffer)
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    loss_ce = nn.BCELoss()
    if args.optimizer == 'SGD':
        baseline_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate, momentum=args.momentum)
    else:
        baseline_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate)
    optimizer = ELLOptimizer(model.parameters(), lr=args.learning_rate, order=args.order, kappa=args.kappa,
                            lambd=args.lambd, exp_bias=args.exp_bias, exp_order=args.exp_order, l1_kappa=args.l1_kappa,
                            theta_grad=args.theta_grad, theta_in_buffer=args.theta_in_buffer,
                            momentum=args.momentum, base_optimizer_type=args.base_optimizer_type, beta1=(0.0 if args.use_rms else 0.9), ell_eps=args.ell_eps, ell_t=args.ell_t)
    print(optimizer.l1_kappa)
    if args.retrain:
        optimizer.init_cache()
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.epochs,  eta_min=args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(baseline_optimizer, factor=1.0, total_iters=1, verbose=True)
    config = {
        "learning_rate" : args.learning_rate,
        "order" : args.order,
        "kappa" : args.kappa,
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
    }
    if args.wandb:
        wandb.init(project=args.project_name, entity="mce", config=config, name=args.run_name)
        wandb.watch(model, log_freq=100)

    historyMaxAcc, historyMaxAccEpoch, maxEpochPeriod = 0.0, 0, 20

    if args.retrain:
        args.epochs = args.epochs + args.retrain_epochs
    # print(args.retrain)
    # print(args.epochs)
    save_dir_base = args.save_dir + "/" + args.run_name

    best_it = {"Test Loss": 10000000.}

    for epoch in range(1, args.epochs + 1):
            print("epoch:{}/{}------------------------------\n".format(epoch, args.epochs))
            # if epoch % 100 == 0:
            #     optimizer.print_weight()
            if epoch == args.epochs - args.retrain_epochs and args.retrain:
                optimizer.prepare_retrain(percentage=0.9)
                optimizer.schedule(new_lr=args.learning_rate)
                baseline_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.retrain_epochs)
            # train :
            dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
            size = len(dataloader.dataset)
            train_loss = 0
            for step, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.reshape(-1, 1).to(device)
                # ty = y.reshape(-1, 1).to(device)
                pred = model(x)
                # print(pred)
                # print(y)
                # print(ty.shape)
                # print(pred.shape)
                loss = loss_ce(pred, y)
                # print(pred.shape)
                # print(y.shape)
                sum_lp_loss = 0.
                if args.lp_penalty:
                    current_lr = scheduler.get_last_lr()[0]
                    lp_penalty_wd = args.lp_penalty_wd / args.learning_rate
                    lp_loss = 0.
                    for i in model.parameters():
                        lp_loss = lp_loss + lp_penalty_wd * torch.abs(i).sum()
                    sum_lp_loss += lp_loss
                    loss = loss + lp_loss
                train_loss += (loss - sum_lp_loss)
                if args.optimizer == 'ell':
                    optimizer.zero_grad()
                else:
                    baseline_optimizer.zero_grad()
                loss.backward()
                if args.optimizer == 'ell':
                    optimizer.step()
                else:
                    baseline_optimizer.step()

                if (step % 100 == 0):
                    loss, current = loss.item(), step * len(x)
                    # wandb.log({"train_loss": loss / (step + 1)})
                    print("loss:{:.6f}, [{}/{}]".format(loss, current, size))

            # scheduler.step()
            train_batches = len(dataloader)
            weighted_sparsity = 0.0
            sparsity_list = [optimizer.sparsity(t) for t in threshold_list]
            for t in threshold_list:
                weighted_sparsity += optimizer.sparsity(t)
            if args.wandb:
                wandb.log({"train_loss": train_loss/train_batches})
                wandb.log({"sparsity": weighted_sparsity})
                for t in threshold_list:
                    wandb.log({"sparsity_threshold_" + str(t): optimizer.sparsity(t)})
            # if epoch == args.epochs:
            #     optimizer.make_sparse()

            test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)
            test_size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            test_loss, correct = 0, 0
            with torch.no_grad():
                for x, y in test_dataloader:
                    x = x.to(device)
                    y = y.reshape(-1, 1).to(device)
                    pred = model(x)
                    loss = loss_ce(pred, y)
                    sum_lp_loss = 0.
                    if args.lp_penalty:
                        current_lr = scheduler.get_last_lr()[0]
                        lp_penalty_wd = args.lp_penalty_wd / args.learning_rate
                        lp_loss = 0.
                        for i in model.parameters():
                            lp_loss = lp_loss + lp_penalty_wd * (torch.abs(i)).sum().detach()
                        sum_lp_loss += lp_loss
                        loss = loss + lp_loss
                    test_loss += (loss - sum_lp_loss)
                    cls_pred = pred > 0.5
                    correct += (cls_pred == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= test_size
            print(f"Train loss: {train_loss/train_batches:>8f}, Test loss: {test_loss:>8f}, Weighted Sparsity: {weighted_sparsity:>8f}, Test Accuracy: {correct * 100 :>8f}\nSparsity List: {str(sparsity_list)}\n")
            for name, p in model.named_parameters():
                print(p.data)

            current_it = {"Train Loss":(train_loss/train_batches).item(), "Test Loss": test_loss.item(), "Weighted Sparsity": weighted_sparsity, "sparsity list": sparsity_list}
            if current_it['Test Loss'] < best_it['Test Loss']:
                best_it = current_it

            if args.wandb:
                wandb.log({"test_acc": 100 * correct})
                wandb.log({"test_loss": test_loss})
            if 100 * correct > historyMaxAcc:
                historyMaxAcc = 100 * correct
                historyMaxAccEpoch = epoch
            #     if not args.tune:
            #         torch.save(model.state_dict(), save_dir_base + "_best.pth")
            # if not args.tune:
            #     torch.save(model.state_dict(), save_dir_base + "_last.pth")
            if args.wandb:
                wandb.log({"lr": scheduler.get_last_lr()[0]})
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if args.optimizer == 'ell':
                optimizer.schedule(new_lr=current_lr)
            if args.wandb:
                wandb.log({"history_max_acc": historyMaxAcc})
            if epoch == args.epochs and args.wandb:
                wandb.log({"final_acc": 100 * correct})
                wandb.log({"final_loss": test_loss})
            if args.tune:
                session.report({"loss": best_it['Test Loss'], "accuracy": historyMaxAcc, "weighted_sparsity": best_it['Weighted Sparsity'], "sparsity":weighted_sparsity, "sparsity_list": sparsity_list})


    print(best_it)
    # if not args.tune :
    #     optimizer.plt_save(save_dir=save_dir_base)

class ELLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, order, kappa, lambd, exp_bias, exp_order, l1_kappa, momentum,
                 beta1=0.9, beta2=0.999, eps=1e-8, theta_grad=False, theta_in_buffer=False,
                 base_optimizer_type='SGD', ell_eps=0.0, ell_t=0):
        self.lr = lr
        self.order = order
        self.kappa = kappa
        self.l1_kappa = l1_kappa
        self.lambd = lambd
        self.ell_eps = ell_eps
        self.exp_bias = exp_bias
        self.ell_t = ell_t
        self.exp_order = 2. * (order - 1) / order
        self.momentum = momentum
        self.momentum_buf = []
        self.exp_avgs = []
        self.exp_avg_sqs = []
        self.init_params = []
        self.denom_buffer = []
        self.retrain_mask = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.current_step = 0
        self.eps = eps
        self.theta_grad = theta_grad
        self.theta_in_buffer = theta_in_buffer
        self.retrain = False
        self.base_optimizer_type = base_optimizer_type
        super().__init__(params, {"lr": lr})

    def init_cache(self):
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                self.init_params.append(p.detach())

    def get_lr(self):
        return self.lr

    def schedule(self, new_lr):
        w = new_lr / self.lr
        self.lr = self.lr * w
        self.kappa = self.kappa * w
        self.l1_kappa = self.l1_kappa * w

    def make_sparse(self, threshold=1e-3):
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                p.data = p.data * (torch.abs(p.data) > threshold)

    def parameter_norm(self, p, truncate_threshold=0.0):
        lp_norm = None
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                # print(param.dtype)
                tmp = param * (torch.abs(param.detach()) > truncate_threshold)
                tmp = tmp.pow(p).sum()
                if lp_norm is None:
                    lp_norm = tmp
                else:
                    lp_norm = lp_norm + tmp
        return torch.pow(lp_norm, 1.0 / float(p))


    def quantile(self, percentage=0.9):
        k = torch.Tensor([])
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                q = torch.abs(p.detach())
                k = k.to(q.device)
                k = torch.concat((k, q.view(-1)), dim=0)
        return torch.quantile(k, percentage, interpolation='lower')

    def plt_save(self, save_dir):
        mpl.rcParams['agg.path.chunksize'] = 10000
        k = torch.Tensor([])
        denom = torch.Tensor([])
        for param_group in self.param_groups:
            params = param_group['params']
            for i, p in enumerate(params):
                q = torch.abs(p.detach())
                k = k.to(q.device)
                k = torch.concat((k, q.view(-1)), dim=0)
                denom = denom.to(q.device)
                denom = torch.concat((denom, self.denom_buffer[i].view(-1)), dim=0)
        kx = k.clone().detach().cpu().numpy()
        denom = denom.cpu().numpy()
        y = np.sort(k.cpu().numpy())
        x = np.arange(len(y))
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(denom)), np.sort(denom))
        plt.yscale('log')
        plt.xlabel('V Rank')
        plt.ylabel('V')
        plt.subplot(2, 1, 2)
        kkx = np.argsort(kx)
        plt.plot(np.arange(len(kkx)), [denom[i] for i in kkx])
        plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel('Parameter Rank')
        plt.ylabel('V')
        plt.savefig(save_dir + "_params.png")

    def get_retrain_mask(self, percentage=0.9):
        mask_threshold = self.quantile(percentage=percentage)
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                q = torch.abs(p) > mask_threshold
                self.retrain_mask.append(q)

    def prepare_retrain(self, percentage=0.9):
        self.get_retrain_mask(percentage=percentage)
        for param_group in self.param_groups:
            params = param_group['params']
            for i, p in enumerate(params):
                p.data = self.init_params[i] * self.retrain_mask[i]
        self.momentum_buf = []
        self.exp_avgs = []
        self.exp_avg_sqs = []
        self.init_params = []
        self.retrain = True


    def step(self, closure=False):
        self.current_step += 1
        for param_group in self.param_groups:
            params = param_group['params']
            tmp_buf = []
            tmp_exp_avgs = []
            tmp_exp_avg_sqs = []
            self.denom_buffer = []


            for i, p in enumerate(params):
                d_p = torch.zeros(p.size())
                p_1_k = torch.pow(torch.abs(p.data), 1. / self.order) * torch.sign(p.data)
                if p.grad is not None:
                    uk = self.order * torch.pow(torch.abs(p_1_k), self.order - 1.)
                    if self.ell_eps <= 0.0:
                        scale = torch.pow(p_1_k, self.ell_t)
                    else:
                        scale = uk / (uk + self.ell_eps)
                    d_p = p.grad * scale
                alpha_ = torch.ones(p.size()).to(p.device)
                if self.theta_grad:
                    if self.theta_in_buffer:
                        d_p = d_p * torch.pow(p_1_k, self.order - 1) * self.order
                    else:
                        alpha_ = alpha_ * torch.pow(p_1_k, self.order - 1) * self.order
                # print()
                res = p_1_k - p_1_k * self.kappa - torch.sign(p_1_k) * self.l1_kappa

                # SGD
                tmp = torch.zeros(p.size())
                if len(self.momentum_buf) > i:
                    tmp = self.momentum_buf[i]

                tmp = tmp.to(d_p.device)
                tmp = tmp * self.momentum + d_p
                tmp_buf.append(tmp)

                # AdamW
                exp_avg = self.exp_avgs[i] if len(self.exp_avgs) > i else torch.zeros_like(p)
                exp_avg_sq = self.exp_avg_sqs[i] if len(self.exp_avg_sqs) > i else torch.zeros_like(p)
                exp_avg = exp_avg.to(d_p.device)
                exp_avg_sq = exp_avg_sq.to(d_p.device)
                exp_avg.mul_(self.beta1).add_(d_p, alpha=1 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(d_p, d_p, value=1 - self.beta2)
                tmp_exp_avgs.append(exp_avg)
                tmp_exp_avg_sqs.append(exp_avg_sq)
                bias_correction1 = 1 - self.beta1 ** self.current_step
                bias_correction2 = 1 - self.beta2 ** self.current_step
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                self.denom_buffer.append(denom - self.eps)
                if self.base_optimizer_type == 'SGD':
                    res.add_(tmp * alpha_, alpha=-self.lr)
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
                    p.data = torch.pow(res, self.order)

            self.momentum_buf = tmp_buf
            self.exp_avgs = tmp_exp_avgs
            self.exp_avg_sqs = tmp_exp_avg_sqs

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

# 1. order 2. penalty
# visible
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim, dim = args.dim, args.dim
        self.predict = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        return F.sigmoid(self.predict(x))