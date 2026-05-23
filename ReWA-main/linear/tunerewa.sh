#!/bin/bash
mkdir -p logs

TASK=linear
OPT=ell
DIM=10000
DLEN=2000
BS=25
EPOCHS=800
LR=0.0002
KAPPA=0.00005
EPS=0.0
SCHED=cosine

for K in 3 5 7 9 11 13; do
    for M in 0 2 4 6 8 10; do
        if [ $M -ge $K ]; then
            continue
        fi
        FNAME="logs/task=${TASK}_opt=${OPT}_K=${K}_M=${M}_dim=${DIM}_dlen=${DLEN}_bs=${BS}_ep=${EPOCHS}_lr=${LR}_kappa=${KAPPA}_eps=${EPS}_sched=${SCHED}.txt"
        python main.py \
            --task $TASK \
            --optimizer $OPT \
            --dim $DIM \
            --dataset_len $DLEN \
            --batch_size $BS \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --order $K \
            --ell_t $M \
            --kappa $KAPPA \
            --ell_eps $EPS \
            --lr_scheduler $SCHED \
            > "$FNAME" 2>&1 
    done
done

wait
echo "All done."