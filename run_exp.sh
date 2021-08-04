#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

for i in 0 2 4
do
    python3 run_exp_local.py -c config/node_gnn$i.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1

for i in 1 3 5
do
    python3 run_exp_local.py -c config/node_gnn$i.yaml &
    sleep 3
done

#export CUDA_VISIBLE_DEVICES=2
#
#for i in 10 11 12 13 14
#do
#    python3 run_exp_local.py -c config/node_gnn$i.yaml &
#    sleep 3
#done
#
#export CUDA_VISIBLE_DEVICES=3
#
#for i in 15 16 17 18 19
#do
#    python3 run_exp_local.py -c config/node_gnn$i.yaml &
#    sleep 3
#done


