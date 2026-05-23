import torch
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
import train
import linear_case
import numpy as np
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def random_search(max_t, num_samples, args):
    # base_lr = 0.19443030226608846
    # base_kappa = 1.39e-06
    # base_lr = 1e-4
    base_lr = 4e-4
    base_kappa = 5e-5
    base_eps = 1e-5
    # base_l1_kappa= 8e-7
    search_params = {
        "learning_rate": tune.grid_search(2 ** (np.arange(5) * 0.5 - 1) * base_lr),
        "kappa": tune.grid_search(2 ** (np.arange(5) * 0.5 - 1) * base_kappa),
        "ell_eps" : tune.grid_search(2 ** (np.arange(5) * 0.5 - 1) * base_eps)
    }
    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=1000,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(linear_case.train_wrapper, args=args),
            resources={"cpu": 2, "gpu": 0.3}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_params,
    )

    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final sparsity: {}".format(best_result.metrics["sparsity"]))