# 基准：K=9 M=2
python main.py \
    --config configs/largescale/resnet18-ell-cifar10.yaml \
    --weight-decay 1e-4 --ell_order 9 --ell_t 2 --ell_eps 0 \
    --name K9_M2 --ell_base_optimizer SGD --seed 42 \
    --multigpu 0 --workers 0

python eval_checkpoint.py \
    --ckpt runs/resnet18-ell-cifar10/K9_M2/ \
    --data 10 --output_csv_name K9_M2.csv --method VanillaConv

# 对比：K=2.25 M=1.25（相同 p ≈ 0.889）
python main.py \
    --config configs/largescale/resnet18-ell-cifar10.yaml \
    --weight-decay 1e-4 --ell_order 2.25 --ell_t 1.25 --ell_eps 0 \
    --name K2.25_M1.25 --ell_base_optimizer SGD --seed 42 \
    --multigpu 0 --workers 0

python eval_checkpoint.py \
    --ckpt runs/resnet18-ell-cifar10/K2.25_M1.25/ \
    --data 10 --output_csv_name K2.25_M1.25.csv --method VanillaConv
