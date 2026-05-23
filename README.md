# ReWA: Reparameterization, Weight Decay, and Adaptive Learning Rate

Official code repository for the ICML 2026 paper:

**"Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate"**

ReWA is a sparse optimization method that reparameterizes the variable as a product of auxiliary variables, combines it with weight decay and a coordinate-wise adaptive learning rate, and provably connects to ℓp (0 < p < 1) regularization. It addresses the optimization instability of direct ℓp methods while achieving stronger sparsity than ℓ1 regularization.

---

## Repository Structure

```
.
├── README.md               ← this file
├── ReWA-main/              ← core algorithm implementation
│   ├── CIFAR10/            ← image classification experiments (CIFAR-10)
│   └── linear/             ← synthetic linear regression experiments
└── ffcv-results-backup/    ← ImageNet training with FFCV

```

---

## Experiments

### Linear Regression (Synthetic)

Please look ReWA-main\linear\README.md

---

### CIFAR-10 / CIFAR-100

Please look ReWA-main\CIFAR10\README.md

---


### ImageNet

ImageNet training uses the [FFCV](https://github.com/libffcv/ffcv/) library. See [`ffcv-results-backup/README.md`](ffcv-results-backup/README.md) for setup. 

Please look ffcv-results-backup\by_eval\README.md



