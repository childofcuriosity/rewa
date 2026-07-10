# ============================================================
# 严格照搬 Spred 原仓库配置 (github.com/zihao-wang/spred)
# Spred: 50 epochs, warmup=1, SpredConv, 扫 wd 9 个值
# L1:   100 epochs, warmup=5, VanillaConv, 扫 l1 5 个值
# ============================================================

echo "===== Spred (原版配置) ====="
for wd in 1e-5 1e-4
do
    python main.py \
        --config configs/largescale/resnet18-spred-cifar10.yaml \
        --weight-decay $wd \
        --name pretrain_weight_decay=$wd \
        --multigpu 0 --workers 0 --seed 42

    python eval_checkpoint.py \
        --ckpt runs/resnet18-spred-cifar10/pretrain_weight_decay=$wd/ \
        --data 10 --output_csv_name spred_wd=$wd.csv --method SpredConv
done

echo "===== L1 (原版配置) ====="
for l1 in 1e-5 1e-4
do
    python main.py \
        --config configs/largescale/resnet18-l1-cifar10.yaml \
        --l1-reg $l1 \
        --name l1=$l1 \
        --multigpu 0 --workers 0 --seed 42

    python eval_checkpoint.py \
        --ckpt runs/resnet18-l1-cifar10/l1=$l1/ \
        --data 10 --output_csv_name l1_reg=$l1.csv --method VanillaConv
done
