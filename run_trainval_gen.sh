#!/bin/sh
for i in 6
do
    python3 gen_train_sampling.py --group_index=$((i))
    python3 gen_val_sampling.py --group_index=$((i))
done
