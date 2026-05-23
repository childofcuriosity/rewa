import copy

import numpy as np
import torch
import csv
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
import argparse
from torch import nn, optim
import random_search
from train import train
import linear_case
import logic_reg
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import wandb

parser = argparse.ArgumentParser(description='random search ELL')

parser.add_argument('--max_t', type=int, default=150)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--project_name', type=str, default="ell_pretrain")
parser.add_argument('--run_name', type=str, default="sample_run")
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--dataset_len', type=int, default=50)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--kappa', type=float, default=0.0)
parser.add_argument('--l1_kappa', type=float, default=0.0)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--order', type=int, default=1)
parser.add_argument('--ell_eps', type=float, default=0.0)
parser.add_argument('--exp_order', type=float, default=2.0)
parser.add_argument('--exp_bias', type=float, default=1.0)
parser.add_argument('--ell_t', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--base_optimizer_type', type=str, default='SGD')
parser.add_argument('--task', type=str, default='CIFAR10')
parser.add_argument('--lambd', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--retrain_epochs', type=int, default=40)
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--theta_grad', action='store_true', default=False)
parser.add_argument('--theta_in_buffer', action='store_true', default=False)
parser.add_argument('--lp_penalty', action='store_true', default=False)
parser.add_argument('--lp_penalty_norm', type=float, default=0.1)
parser.add_argument('--lp_penalty_wd', type=float, default=1e-5)
parser.add_argument('--lp_penalty_th', type=float, default=1e-3)
parser.add_argument('--retrain_percentage', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=9913)
parser.add_argument('--backbone', type=str, default="resnet18")
parser.add_argument('--test_seed', type=int, default=1113)
parser.add_argument('--wandb', action='store_true', default=False)
parser.add_argument('--tune', action='store_true', default=False)
parser.add_argument('--update_kappa', action='store_true', default=False)
parser.add_argument('--lr_scheduler', default='cosine', help='lr_scheduler')
parser.add_argument('--use_rms', action='store_true', default=False)
parser.add_argument('--kappa_scheduler', action='store_true', default=False)
parser.add_argument('--schedule_order', action='store_true', default=False)
parser.add_argument('--kappa_scheduler_start_epoch', type=int, default=100)
parser.add_argument('--kappa_scheduler_zero', action='store_true', default=False)
parser.add_argument('--lr_min', default=0.0, type=float, help="min learning rate")
parser.add_argument('--save_dir', type=str, default="./output")


def read_csv_without_header(csv_file_path):
    data = []

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader, None)

        for row in reader:
            data.append(row)

    return data


def compute_mean_and_std_per_position(*args):
    lengths = [len(lst) for lst in args]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All input lists must have the same length.")

    arrays = [np.array(lst) for lst in args]
    mean_values = np.mean(arrays, axis=0)

    std_values = np.std(arrays, axis=0, ddof=1)

    return mean_values, std_values

if __name__ == '__main__':

    args = parser.parse_args()
    if args.tune:
        random_search.random_search(args.max_t, args.num_samples, args)
    elif args.task == 'linear':
        linear_case.train(args)
    elif args.task == 'logic':
        logic_reg.train(args)
    else:
        train(args)

