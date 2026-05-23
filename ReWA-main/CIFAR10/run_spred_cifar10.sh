for wd in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    CUDA_VISIBLE_DEVICES=3 python3 main.py --config configs/largescale/resnet18-ell-cifar10.yaml \
                    --weight-decay $wd \
                    --ell_order 3 \
                    --name new_ell_order=1-weight_decay=$wd \
                    --multigpu 0
done