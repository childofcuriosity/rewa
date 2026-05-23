rm -rf fast_runs
fast_spred() {
    python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                    --weight-decay $1 \
                    --name fast_run_ell9_1e-4__wd=$1 \
                    --multigpu $2 \
                    --pretrained runs/resnet18-ell-cifar10/ell_order\=9-weight_decay\=1e-4/0/checkpoints/model_best.pth \
                    --log-dir fast_runs \
                    --save_every 100

}

fast_spred 1e-5 2 &
fast_spred 1e-4 3 &

wait

fast_spred 1e-3 2 &
fast_spred 1e-2 3 &

wait

fast_spred 3e-5 2 &
fast_spred 3e-4 3 &

wait

fast_spred 3e-3 2 &
fast_spred 3e-2 3 &

wait