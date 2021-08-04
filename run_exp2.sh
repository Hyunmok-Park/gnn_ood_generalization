#!/bin/sh
export CUDA_VISIBLE_DEVICES=1

for i in 13
do
    python3 run_exp_local.py -c config/node_gnn$i.yaml
    sleep 3
done

#export CUDA_VISIBLE_DEVICES=2
#for i in 1 2
#do
#    python3 run_exp_local.py -c config/node_gnn$i.yaml
#    sleep 3
#done


