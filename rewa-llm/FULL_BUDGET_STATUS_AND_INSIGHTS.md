# ReWA-LLM 全预算调参：情况、证据与下一步

更新日期：2026-09-24
工作分支：`organize-project`

## 结论摘要

现有结果显示，训练预算是当前 ReWA-LLM 实验的首要控制变量。相同的
`K=9, M=2, lr=0.006, epsilon=0, rewa_weight_decay=1e-4` 配置，在长程
训练中从 6.55M tokens 时的验证 PPL 12.84 持续下降到 72.09M tokens 时
的 PPL 6.37。此前用于剪枝比较的 ReWA checkpoint 只训练了 10M tokens，
因此原来的 70%/80% 剪枝曲线同时包含了“方法的可剪枝性”和“模型尚未充分
收敛”两种影响。

强 L1 仍是需要超过的直接基线。历史强 L1 checkpoint 使用
`alpha=1e-5`，其 PPL 为 13.95（未剪枝）、13.97（70%）和 14.69
（80%）。它牺牲了一部分未剪枝精度，却训练出了对一次性 magnitude
pruning 高度稳定的工作点。新的协议将 ReWA 和强 L1 都训练 100M tokens，
用同一验证集、seed、有效 batch 和剪枝评估口径比较。

本机实验已按要求暂停。训练进程、L1 等待队列和 GPU 任务均已停止；本地
checkpoint 完整保留，但不进入 Git。

## 已观察到的结果

### 长程 ReWA 锚点

配置：

| 项目 | 值 |
| --- | ---: |
| 模型 | 6-layer LLaMA-style decoder，约 14M 参数 |
| 数据 | TinyStories，100,000 train / 5,000 validation documents |
| 训练 tokens | 目标 100M |
| 有效 batch | 128（`batch_size=32`, `gradient_accumulation=4`） |
| ReWA | `K=9, M=2, epsilon=0` |
| peak LR | `0.006` |
| y-space weight decay | `1e-4` |
| scheduler | 100-step warmup，cosine decay 到 0 |
| seed | 0 |

固定验证批次上的曲线：

| Step | Tokens | Token-equivalent epochs | Validation PPL |
| ---: | ---: | ---: | ---: |
| 200 | 6.55M | 0.30 | 12.8357 |
| 400 | 13.11M | 0.61 | 9.7230 |
| 600 | 19.66M | 0.91 | 8.7474 |
| 800 | 26.21M | 1.21 | 8.0986 |
| 1000 | 32.77M | 1.52 | 7.7092 |
| 1200 | 39.32M | 1.82 | 7.4114 |
| 1400 | 45.88M | 2.12 | 7.1321 |
| 1600 | 52.43M | 2.43 | 6.8847 |
| 1800 | 58.98M | 2.73 | 6.6814 |
| 2000 | 65.54M | 3.03 | 6.5082 |
| 2200 | 72.09M | 3.33 | **6.3683** |

验证 PPL 在所有已完成验证点上单调下降。step 2200 是暂停时保存的
`latest.pt` 和 `best.pt`，训练目标 3052 steps 尚未完成，100M-token
checkpoint 也尚未做 global-pruning 评估。

### 历史稀疏结果

| 方法与 checkpoint | 训练 tokens | PPL 0% | PPL 70% | PPL 80% |
| --- | ---: | ---: | ---: | ---: |
| 强 L1，`alpha=1e-5` | 20M | 13.9520 | **13.9731** | **14.6939** |
| ReWA K9/M2，`lr=0.006, wd=1e-4, eps=0` | 10M | **10.9777** | 18.2948 | 43.4725 |

这两条曲线揭示了不同的 Pareto 工作点：10M ReWA 保留了更好的未剪枝
精度，并显著超过普通 Dense/弱 L1 的高稀疏曲线；强 L1 直接针对最终
`|x|` magnitude pruning 塑造权重，在 70%/80% 时保持了更低的绝对 PPL。

## Insights

### 1. 原 10M/20M 比较低估了充分训练的 ReWA

100M 锚点在 20M tokens 之后仍持续改善，说明旧实验的 token budget
不足以隔离优化器与剪枝结构的作用。下一轮所有候选都使用完整 100M tokens，
并按固定 validation batches 选择 `best.pt`；不再用 1M/3M 的短预算表现
淘汰配置。

### 2. 当前 `rewa_weight_decay=1e-4` 是弱显式正则

100M 调度的精确累计学习率为 `sum(lr_t)=9.162`。只考虑
`y <- (1-lr_t*wd)y` 的纯衰减时，各候选的累计因子为：

| y-space wd | y 保留比例 | 映射为 K=9 后的 x 保留比例 |
| ---: | ---: | ---: |
| `1e-4` | 0.99908 | 0.99179 |
| `1e-2` | 0.91245 | 0.43841 |
| `3e-2` | 0.75966 | 0.08426 |
| `1e-1` | 0.39995 | 0.000262 |
| `3e-1` | 0.06390 | 1.78e-11 |

这项计算隔离的是显式 decay；真实训练还包含任务梯度和 Adam moments。
它说明现有 `1e-4` 几乎是无衰减锚点，而 `1e-2` 到 `1e-1` 覆盖了真正
改变 K9 权重尺度的区域。新网格因此在对数尺度上搜索
`{1e-4, 1e-2, 3e-2, 1e-1, 3e-1}`。

### 3. epsilon 是近零区域的动力学开关

实现中的正 epsilon 通过
`J/(J+epsilon)`（`J=K|y|^(K-1)`）衰减近零坐标的梯度。它可能扩大
近零权重群、提高 magnitude-pruning 稳定性，也可能冻结仍有任务贡献的
小权重。第一轮使用 `{0, 1e-6, 1e-4, 1e-3, 1e-2}` 与 weight decay
做完整 Cartesian 搜索。

`M=2, epsilon>0` 位于论文 Configuration-B 严格范围之外。runner 会把
这些点明确标记为 empirical boundary probe；它们用于回答实际 LLM
优化问题，不作为该理论配置的验证结果。

### 4. L1 与剪枝目标直接对齐，ReWA 需要联合调参

L1 训练目标直接包含 `alpha * sum(|x|)`，剪枝也按 `|x|` 排序，因此
`alpha=1e-5` 能让模型在训练期间适应大量弱连接。ReWA 的稀疏偏置来自
`x=sign(y)|y|^K`、梯度缩放、epsilon 和 y-space decay 的共同作用。
单独提高 LR 解决了早期停滞，但尚未系统探索决定最终近零结构的另外两个
轴。`LR × epsilon × decay` 的相互作用是下一阶段的核心量。

### 5. 未剪枝 PPL 与高稀疏 PPL 必须由同一个 checkpoint 报告

最终汇总器按同一个 checkpoint 在 70% 和 80% 的最坏相对 L1 比率排序，
不允许为不同 sparsity 从不同运行挑选点。成功标准为同一 ReWA checkpoint
同时满足：

- 70% PPL `< 13.9731`；
- 80% PPL `< 14.6939`；
- 未剪枝 PPL 保持在可接受范围；
- 之后再用完整 100M-token 训练扩展 seeds。

## 已冻结的下一轮协议

运行入口：

```bash
PYTHON=.venv/bin/python \
BATCH_SIZE=32 GRAD_ACCUM=4 \
bash scripts/run_full_budget_round1.sh
```

该脚本依次执行：

1. 100M-token 强 L1（`alpha=1e-5`）匹配预算基线；
2. K9/M2、`lr=0.006` 下 5 个 epsilon × 5 个 weight decay，共 25 个
   完整训练配置；
3. `(K,M) = (3,0), (3,1), (5,0), (5,2), (7,2), (9,2), (9,4)`
   七个 100M-token 几何对照；
4. 对每个 checkpoint 用相同 100 validation batches 评估
   0%/50%/70%/80% global sparsity；
5. 生成 `full-budget-comparison.csv`、`FULL_BUDGET_RESULTS.md`、PNG 和
   PDF 对比图。

第一轮完成后，从完整训练结果识别有潜力区域，再以完整 100M-token 预算
细化学习率，并对领先配置扩展 seed。短训练曲线只用于诊断稳定性，不用于
候选淘汰。

## 暂停与恢复

Git 分支不包含数据、输出或 checkpoint。fresh clone 会从头执行完整协议。

本机可恢复目录：

```text
outputs/rewa-full-budget/
artifacts/rewa-full-budget/tuning-manifest.csv
```

checkpoint：

| 文件 | 状态 | SHA256 |
| --- | --- | --- |
| `.../k9-m2-eps0-wd1e-4-lr0.006-seed0/latest.pt` | step 2200 resume state | `3CBC736368D8EA69ACF2D8667E2C51C43AED87BC648DE9C279A1186F320E2854` |
| `.../k9-m2-eps0-wd1e-4-lr0.006-seed0/best.pt` | step 2200 validation best | `0491FB726D5ABC42ABF86B7C40498974ABF81408FD32DBD805085258181C5A2B` |

将上述输出目录和 manifest 一并复制到新设备后，重新执行相同 runner 会从
`latest.pt` 恢复。只从 GitHub clone 时，所有配置会从 seed 0 的初始状态
重新训练。

## 当前未完成项

- 100M ReWA 锚点还剩 852 steps；
- 100M 强 L1 匹配预算基线尚未运行；
- 100M ReWA checkpoint 尚未做 70%/80% pruning；
- epsilon × decay、K/M 和后续 LR 细化网格尚未运行；
- 多 seed 确认、最终表格和最终图尚未生成。

因此，当前证据支持“长训练显著改善 ReWA 的未剪枝收敛”以及“显式正则
仍有大范围未探索空间”。ReWA 是否能在同预算下同时超过强 L1 的 70% 和
80% PPL，将由上述完整网格直接检验。
