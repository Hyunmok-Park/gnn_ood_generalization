#!/bin/sh

for i in $(seq 16)
do
    python3 list_gen.py --ii=$((i-1)) &
done

#export CUDA_VISIBLE_DEVICES=1
#
#for i in 2
#do
#    python3 run_exp_local.py -c config/node_gnn$i.yaml &
#    sleep 3
#done
