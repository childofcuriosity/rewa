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
# import wandb
# from dataset import GaussianDataset
# from model.resnet import ResNet20
# from model.resnet20 import resnet20


# parser = argparse.ArgumentParser(description='random search ELL')
#
# # parser.add_argument('--max-t', type=int, default=10)
# # parser.add_argument('--num-samples', type=int, default=400)
# parser.add_argument('--learning_rate', type=float, default=0.01)
# parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--dataset_len', type=int, default=50)
# parser.add_argument('--dim', type=int, default=100)
# parser.add_argument('--kappa', type=float, default=0.0)
# parser.add_argument('--order', type=int, default=1)
# parser.add_argument('--exp_order', type=float, default=2.0)
# parser.add_argument('--exp_bias', type=float, default=1.0)
# parser.add_argument('--lambd', type=float, default=0.0)

def model_sparsity(model, threshold):
    cnt = 0
    tot_cnt = 0
    for p in model.parameters():
        cnt += (p.data > threshold).sum().item() + (p.data < -threshold).sum().item()
        tot_cnt += torch.ones(p.size()).sum().item()
    return 1. - float(cnt) / float(tot_cnt)

# def train_wrapper(config, args):
#     args.__dict__.update(config)
#     tmp_name = "tune-"
#     for k, v in config.items():
#         tmp_name = tmp_name + str(k) + "=" + str(v)
#     args.run_name = tmp_name
#     train(args)
#
# def train(args):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     if args.backbone == "resnet50":
#         trans = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])
#     else:
#         trans = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(32, 4),
#             transforms.ToTensor(),
#             normalize
#         ])
#     test_trans = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
#     seed = args.seed
#     test_seed = args.test_seed
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     threshold_list = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 1e-3]
#     # training_data = GaussianDataset(seed=seed)
#     # test_data = GaussianDataset(seed=test_seed)
#     if args.task == 'CIFAR10':
#         training_data = datasets.CIFAR10(root="~/ell/data", train=True, download=True, transform=trans)
#         test_data = datasets.CIFAR10(root="~/ell/data", train=False, download=True, transform=test_trans)
#     elif args.task == 'CIFAR100':
#         training_data = datasets.CIFAR100(root="~/ell/data", train=True, download=True, transform=trans)
#         test_data = datasets.CIFAR100(root="~/ell/data", train=False, download=True, transform=test_trans)
#     else:
#         raise NotImplementedError
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#     model = Network(args)
#     model.to(device)
#     # print(args.theta_in_buffer)
#     # for name, param in model.named_parameters():
#     #     print(name, param.size())
#     loss_ce = nn.CrossEntropyLoss()
#     if args.optimizer == 'SGD':
#         baseline_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate, momentum=args.momentum)
#     else:
#         baseline_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate)
#
#     optimizer = ELLOptimizer(model.parameters(), lr=args.learning_rate, order=args.order, kappa=args.kappa,
#                             lambd=args.lambd, exp_bias=args.exp_bias, exp_order=args.exp_order, l1_kappa=args.l1_kappa,
#                             theta_grad=args.theta_grad, theta_in_buffer=args.theta_in_buffer,
#                             momentum=args.momentum, base_optimizer_type=args.base_optimizer_type, beta1=(0.0 if args.use_rms else 0.9),
#                              update_kappa=args.update_kappa)
#     if args.retrain:
#         optimizer.init_cache()
#     if args.lr_scheduler == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.epochs,  eta_min=args.lr_min)
#     elif args.lr_scheduler == 'MultiStep':
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                             milestones=[100, 150])
#     else:
#         scheduler = torch.optim.lr_scheduler.ConstantLR(baseline_optimizer, factor=1.0, total_iters=1, verbose=True)
#     config = {
#         "learning_rate" : args.learning_rate,
#         "order" : args.order,
#         "kappa" : args.kappa,
#         "epochs" : args.epochs,
#         "batch_size" : args.batch_size,
#     }
#     if args.wandb:
#         wandb.init(project=args.project_name, entity="mce", config=config, name=args.run_name)
#         wandb.watch(model, log_freq=100)
#
#     historyMaxAcc, historyMaxAccEpoch, maxEpochPeriod = 0.0, 0, 20
#
#     if args.retrain:
#         args.epochs = args.epochs + args.retrain_epochs
#     print(args.retrain)
#     print(args.epochs)
#     save_dir_base = args.save_dir + "/" + args.run_name
#
#     for epoch in range(1, args.epochs + 1):
#             print("epoch:{}/{}------------------------------\n".format(epoch, args.epochs))
#             log_dict = {}
#             # if epoch % 100 == 0:
#             #     optimizer.print_weight()
#             if epoch == args.epochs - args.retrain_epochs and args.retrain:
#                 optimizer.prepare_retrain(percentage=0.9)
#                 optimizer.schedule(new_lr=args.learning_rate)
#                 baseline_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.kappa / args.learning_rate)
#                 scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.retrain_epochs)
#
#             # train :
#             dataloader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
#             size = len(dataloader.dataset)
#             train_loss = 0
#             log_dict['train_loss'] = 0.
#             log_dict['train_lp_loss'] = 0.
#             for step, (x, y) in enumerate(dataloader):
#                 x = x.to(device)
#                 y = y.to(device)
#                 # ty = y.reshape(-1, 1).to(device)
#                 pred = model(x)
#                 # print(pred)
#                 # print(y)
#                 # print(ty.shape)
#                 # print(pred.shape)
#                 loss = loss_ce(pred, y)
#                 log_dict['train_loss'] = log_dict['train_loss'] + loss
#                 origin_loss = loss
#                 # print(loss.item())
#                 if args.lp_penalty:
#                     current_lr = scheduler.get_last_lr()[0]
#                     lp_penalty_wd = args.lp_penalty_wd / args.learning_rate
#                     log_dict['lp_wd'] = lp_penalty_wd
#                     lp_loss = 0.
#                     for i in model.parameters():
#                         lp_loss = lp_loss + lp_penalty_wd * torch.abs(i).sum()
#                     loss = loss + lp_loss
#                     log_dict['train_lp_loss'] = log_dict['train_lp_loss'] + lp_loss
#                     # print(loss.item())
#                     # loss += args.lp_penalty_wd * optimizer.parameter_norm(args.lp_penalty_norm, args.lp_penalty_th)
#
#                 if args.optimizer == 'ell':
#                     optimizer.zero_grad()
#                 else:
#                     baseline_optimizer.zero_grad()
#                 loss.backward()
#                 if args.optimizer == 'ell':
#                     optimizer.step()
#                 else:
#                     baseline_optimizer.step()
#
#                 if (step % 100 == 0):
#                     loss, current = loss.item(), step * len(x)
#                     train_loss += loss
#                     # wandb.log({"train_loss": loss / (step + 1)})
#                     print("loss:{:.6f}, [{}/{}]".format(loss, current, size))
#
#             # scheduler.step()
#             train_batches = len(dataloader)
#             weighted_sparsity = 0.0
#             sparsity_list = [model_sparsity(model, t) for t in threshold_list]
#             for t in threshold_list:
#                 weighted_sparsity += optimizer.sparsity(t)
#             if args.wandb:
#                 log_dict['train_loss'] = log_dict['train_loss'] / train_batches
#                 log_dict['train_lp_loss'] = log_dict['train_lp_loss'] / train_batches
#                 log_dict['sparsity'] = weighted_sparsity
#
#                 # wandb.log({"train_loss": train_loss/train_batches})
#                 # wandb.log({"sparsity": weighted_sparsity})
#                 for t in threshold_list:
#                     # wandb.log({"sparsity_threshold_" + str(t): model_sparsity(model, t)})
#                     log_dict['sparsity_threshold_'+str(t)] = model_sparsity(model, t)
#             # if epoch == args.epochs:
#             #     optimizer.make_sparse()
#
#             test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)
#             test_size = len(test_dataloader.dataset)
#             num_batches = len(test_dataloader)
#             test_loss, correct, sum_lp_loss = 0, 0, 0
#             with torch.no_grad():
#                 for x, y in test_dataloader:
#                     x = x.to(device)
#                     y = y.to(device)
#                     pred = model(x)
#                     loss = loss_ce(pred, y)
#                     if args.lp_penalty:
#                         current_lr = scheduler.get_last_lr()[0]
#                         lp_penalty_wd = args.lp_penalty_wd / args.learning_rate
#                         lp_loss = 0.
#                         for i in model.parameters():
#                             lp_loss = lp_loss + lp_penalty_wd * (torch.abs(i.detach()) * i.detach()).sum()
#                         sum_lp_loss += lp_loss
#                         loss = loss + lp_loss
#                     test_loss += loss
#
#                     correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#             test_loss /= num_batches
#             sum_lp_loss /= num_batches
#             correct /= test_size
#             print(f"Train loss: {train_loss:>8f}, Test loss: {test_loss:>8f}, Weighted Sparsity: {weighted_sparsity:>8f}, Test Accuracy: {correct * 100 :>8f}\n")
#             if args.wandb:
#                 # wandb.log({"test_acc": 100 * correct})
#                 # wandb.log({"test_loss": test_loss})
#                 log_dict['test_acc'] = 100 * correct
#                 log_dict['test_loss'] = test_loss - sum_lp_loss
#                 log_dict['test_lp_loss'] = sum_lp_loss
#             if 100 * correct > historyMaxAcc:
#                 historyMaxAcc = 100 * correct
#                 historyMaxAccEpoch = epoch
#                 if not args.tune:
#                     torch.save(model.state_dict(), save_dir_base + "_best.pth")
#             if not args.tune:
#                 torch.save(model.state_dict(), save_dir_base + "_last.pth")
#             if args.wandb:
#                 # wandb.log({"lr": scheduler.get_last_lr()[0]})
#                 log_dict['lr'] = scheduler.get_last_lr()[0]
#             scheduler.step()
#             current_lr = scheduler.get_last_lr()[0]
#             if args.optimizer == 'ell':
#                 optimizer.schedule(new_lr=current_lr)
#                 log_dict['kappa'] = optimizer.kappa
#                 if args.kappa_scheduler and epoch >= args.kappa_scheduler_start_epoch:
#                     if args.kappa_scheduler_zero:
#                         optimizer.schedule_kappa(0.)
#                     else:
#                         optimizer.schedule_kappa(current_lr / args.learning_rate * args.kappa)
#                     if epoch == args.kappa_scheduler_start_epoch and args.schedule_order:
#                         optimizer.update_order(1)
#
#             if args.wandb:
#                 # wandb.log({"history_max_acc": historyMaxAcc})
#                 log_dict['history_max_acc'] = historyMaxAcc
#             if epoch == args.epochs and args.wandb:
#                 # wandb.log({"final_acc": 100 * correct})
#                 # wandb.log({"final_loss": test_loss})
#                 log_dict['final_acc'] = 100 * correct
#                 log_dict['final_loss'] = test_loss
#             if args.wandb:
#                 wandb.log(log_dict)
#             if args.tune:
#                 session.report({"loss": test_loss, "accuracy": historyMaxAcc, "weighted_sparsity": weighted_sparsity, "sparsity":weighted_sparsity, "sparsity_list": sparsity_list})
#
#     if not args.tune:
#         optimizer.plt_save(save_dir=save_dir_base)
#
#
    #Todo: weight decay, momentum, adam
class ELLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, order=1, kappa=0.0, lambd=0.0, exp_bias=0.0, exp_order=0.0, l1_kappa=0.0, momentum=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8, theta_grad=False, theta_in_buffer=False,
                 base_optimizer_type='SGD', update_kappa=False,ell_eps=0.0, ell_t=0, t_in_eta=False, clip_t=False, eps_scale=False,
                ):
        self.lr = lr
        self.order = order
        self.kappa = kappa
        self.l1_kappa = l1_kappa # 这个好像用不到
        self.lambd = lambd
        self.exp_bias = exp_bias
        self.exp_order = 2. * (order - 1) / order
        self.momentum = momentum
        self.momentum_buf = []
        self.exp_avgs = []
        self.ell_eps = ell_eps
        self.ell_t = ell_t
        self.t_in_eta = t_in_eta
        self.clip_t = clip_t
        self.exp_avg_sqs = []
        self.init_params = []
        self.denom_buffer = []
        self.retrain_mask = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.current_step = 0
        self.eps = eps
        self.eps_scale = eps_scale
        self.theta_grad = theta_grad
        self.theta_in_buffer = theta_in_buffer
        self.retrain = False
        self.base_optimizer_type = base_optimizer_type
        self.update_kappa = update_kappa
        super().__init__(params, {"lr": lr})


    def update_order(self, new_order):
        self.order = new_order

    def init_cache(self):
        for param_group in self.param_groups:
            params = param_group['params']
            for p in params:
                self.init_params.append(p.detach())

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
            lr = param_group['lr']
            kappa = param_group['weight_decay']
            params = param_group['params']
            # ★ NEW: 按 base_optimizer_type 初始化状态
            if self.base_optimizer_type == 'SGD':
                momentum_buf = []
                if 'momentum_buf' in param_group:
                    momentum_buf = param_group['momentum_buf']
                current_buf = []
            else:  # AdamW
                if 'exp_avg' not in param_group:
                    param_group['exp_avg'] = [torch.zeros_like(p.data) for p in params]
                    param_group['exp_avg_sq'] = [torch.zeros_like(p.data) for p in params]

            for i, p in enumerate(params):
                d_p = torch.zeros(p.size())
                p_1_k = torch.pow(torch.abs(p.data), 1. / self.order) * torch.sign(p.data)
                alpha_ = torch.ones(p.size()).to(p.device)
                if p.grad is not None:
                    # d_p = torch.renorm(p.grad.reshape(1, -1), 2, 1, 1).reshape(-1)
                    uk = self.order * torch.pow(torch.abs(p_1_k), self.order - 1.)

                    # 统一 scale 计算（Algorithm 1 & 2 公式相同）
                    if self.ell_eps <= 0.0:
                        scale = torch.pow(p_1_k, self.ell_t)
                    else:
                        scale = uk / (uk + self.ell_eps) * torch.pow(p_1_k, self.ell_t)

                    if self.clip_t:
                        scale = torch.clamp(scale, min=-1, max=1)

                    d_p = p.grad
                    if self.t_in_eta:
                        alpha_ = alpha_ * scale
                    else:
                        d_p = d_p * scale

                if self.theta_grad:
                    if self.theta_in_buffer:
                        d_p = d_p * torch.pow(p_1_k, self.order - 1) * self.order
                    else:
                        alpha_ = alpha_ * torch.pow(p_1_k, self.order - 1) * self.order


                if self.base_optimizer_type == 'SGD':
                    # --- Algorithm 1: SGD + Momentum（原有逻辑不变）---
                    d_p += p_1_k * kappa + torch.sign(p_1_k) * self.l1_kappa
                    res = p_1_k

                    tmp = torch.zeros(p.size())
                    if len(momentum_buf) > i:
                        tmp = momentum_buf[i]
                    tmp = tmp.to(d_p.device)
                    tmp = tmp * self.momentum + d_p
                    current_buf.append(tmp)

                    res.add_(tmp * alpha_, alpha=-lr)

                else:
                    # ★ NEW: Algorithm 2: AdamW
                    # 1) Decoupled weight decay
                    res = p_1_k * (1.0 - lr * kappa)
                    # if self.l1_kappa > 0:
                    #     res = res - lr * torch.sign(p_1_k) * self.l1_kappa

                    # 2) Replaced gradient (合并 d_p 和 alpha_)
                    g = (d_p * alpha_).to(p.device)

                    # 3) Adam moment updates
                    exp_avg = param_group['exp_avg'][i]
                    exp_avg_sq = param_group['exp_avg_sq'][i]

                    exp_avg.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                    exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                    # 4) Bias correction
                    bc1 = 1.0 - self.beta1 ** self.current_step
                    bc2 = 1.0 - self.beta2 ** self.current_step

                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(self.eps)

                    # 5) Adam step on y
                    res.addcdiv_(exp_avg, denom, value=-step_size)
                # SGD

                # print(p)
                # print(torch.norm(p_1_k) ** 2)
                # print(p.shape)
                # print(p_1_k.shape)
                # p.data = torch.pow(p_1_k - p_1_k * self.kappa - torch.sign(p_1_k) * self.l1_kappa - self.lr * tmp, self.order)
                if self.retrain:
                    p.data = torch.pow(res * self.retrain_mask[i], self.order)
                else:
                    p.data = torch.pow(res, self.order)

            param_group['momentum_buf'] = current_buf
            # self.momentum_buf = tmp_buf

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
# class Network(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         # self.conv1 = nn.Conv2d(3, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16*5*5, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#         # self.backbone = torchvision.models.resnet18(pretrained=False)
#
#         self.backbone = resnet20()
#         # if args.backbone == "resnet18":
#         #     self.backbone = torchvision.models.resnet18(pretrained=args.pretrain)
#         #     hidden_dim = 512
#         # elif args.backbone == "resnet50":
#         #     self.backbone = torchvision.models.resnet50(pretrained=args.pretrain)
#         #     hidden_dim = 2048
#         # else:
#         #     raise NotImplementedError
#         #
#         # if args.backbone == "resnet18":
#         #     self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
#         #     self.backbone.maxpool = nn.MaxPool2d(1, 1, 0)
#         #
#         # if args.task == 'CIFAR10':
#         #     self.backbone.fc = nn.Linear(hidden_dim, 10)
#         # elif args.task == 'CIFAR100':
#         #     self.backbone.fc = nn.Linear(hidden_dim, 100)
#         # else:
#         #     raise NotImplementedError
#
#         # self.backbone = nn.Linear(100, 100)
#         # self.predict = nn.Linear(100, 1)
#
#     def forward(self, x):
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = x.view(-1, 16 * 5 * 5)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         return self.backbone(x)
#         # return self.predict(F.relu(self.backbone(x)))