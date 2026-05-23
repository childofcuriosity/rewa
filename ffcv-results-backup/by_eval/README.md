# ReWA 

## Requirements

As ffcv. Please look ffcv-results-backup\README.md

## Usage



`python train_imagenet.py --config-file rn50_configs/rn50_88_epochs.yaml --data.train_dataset=<your train_dataset> --data.val_dataset=<your val_dataset> <your args>`

`python retrain_imagenet.py --config-file rn50_configs/rn50_88_epochs.yaml --data.train_dataset=<your train_dataset> --data.val_dataset=<your train_dataset> --model.load_from=<your model checkpoint> --model.load_percent=<your model load_percent> <your args>`

Some hyperparameters: 
- `--lr.lr`: learning rate
- `--training.order`: K
- `--training.ell_m`: M
- `--training.ell_eps`: epsilon
- `--training.weight_decay`: weight decay


You can look ffcv-results-backup\by_eval\ReWA_ablation\train_all.sh and ffcv-results-backup\by_eval\retrain_all_fix.sh for raw usages.





