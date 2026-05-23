python3 main.py --config configs/largescale/resnet18-ell-cifar10.yaml \
    --weight-decay 1e-4 \
    --ell_eps 0\
    --ell_order 9 \
    --ell_t 2 \
    --name new_ell_order=9-weight_decay=1e-4_t=2_eps=0_adam \
    --multigpu 0 \
    --save_every 100
    --ell_base_optimizer AdamW
python eval_checkpoint.py \
  --ckpt runs/resnet18-ell-cifar10/new_ell_order=9-weight_decay=1e-4_t=2_eps=0_adam/ \
  --data 10 \
  --output_csv_name cifar10_ell9_wd1e-4_t2.csv \
  --method VanillaConv