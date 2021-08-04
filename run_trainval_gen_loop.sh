#!/bin/sh
python3 gen_train_sampling_loop.py --group_index=$((i))
python3 gen_val_sampling_loop.py --group_index=$((i))

