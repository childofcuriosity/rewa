for ell_t in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2 python3 main.py --config configs/largescale/resnet18-ell-cifar10.yaml \
                    --weight-decay 1e-4 \
                    --ell_t $ell_t\
                    --ell_order 9 \
                    --name new_ell_order=9-weight_decay=1e-4_t=$ell_t \
                    --multigpu 0 \
                    --save_every 100
done