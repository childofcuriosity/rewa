# ReWA on Linear Task 

## Requirements

- Conda
- Python > 3.7

You can create ReWA environment with cmd `conda env create -f environment.yaml`

## Usage

`python main.py --task linear --batch_size 25 --epochs 1000 --optimizer ell --lr_scheduler cosine --dim 10000 --dataset_len 2000 <your args>`

Some hyperparameters: 

- `--order`: K
- `--ell_t`: M
- `--ell_eps`: epsilon
- `--kappa`: weight decay






