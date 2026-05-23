#python3 eval_checkpoint.py --ckpt runs/resnet18-spred-cifar10/pretrain_weight_decay\=1e-4_sgd_new/ --data 10 --output_csv_name cifar10_spred.csv --method 0
#python3 eval_checkpoint.py --ckpt runs/resnet18-ell-cifar10/new_ell_order\=3-weight_decay\=1e-4/ --data 10 --output_csv_name cifar10_ell_3.csv
python3 eval_checkpoint.py --ckpt runs/resnet18-ell-t-cifar10/new_ell_order=9-weight_decay=1e-4_t=2/ --data 10 --output_csv_name cifar10_ell_9_t_new.csv
#python3 eval_checkpoint.py --ckpt runs/resnet18-l1-cifar10/l1\=1e-5 --data 10 --output_csv_name cifar10_l1.csv
#python3 eval_checkpoint.py --ckpt runs/resnet18-ell-t-cifar10/ --data 10 --output_csv_name cifar10_ell_9_t.csv
#python3 eval_checkpoint.py --ckpt runs/resnet18-ell-eps-cifar10/ --data 10 --output_csv_name cifar10_ell_9_eps.csv
#python3 eval_checkpoint.py --ckpt fast_runs/resnet18-spred-cifar10/fast_run_ell9_3e-5__wd\=1e-4/ --data 10 --output_csv_name cifar10.csv --method 0
#python3 eval_checkpoint.py --ckpt runs/resnet18-spred-cifar100 --data 100 --output_csv_name cifar100.csv --method 0
