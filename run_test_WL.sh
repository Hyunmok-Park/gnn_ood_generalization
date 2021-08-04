#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

for i in $(seq 5)
do
    python3 run_test_WL.py --index=$((i-1)) &
    sleep 1
done

#export CUDA_VISIBLE_DEVICES=1
#
#for i in $(seq 13)
#do
#    python3 run_test_WL.py --index=$((i-1 + 12)) &
#    sleep 1
#done

######################

#export CUDA_VISIBLE_DEVICES=0
#
#for i in $(seq 13)
#do
#    python3 run_test_WL.py --index=$((i-1 + 26)) &
#    sleep 1
#done
#
#export CUDA_VISIBLE_DEVICES=1
#
#for i in $(seq 13)
#do
#    python3 run_test_WL.py --index=$((i-1 + 39)) &
#    sleep 1
#done

#######################

#export CUDA_VISIBLE_DEVICES=0
#
#for i in $(seq 13)
#do
#    python3 run_test_WL.py --index=$((i-1 + 52)) &
#    sleep 1
#done

#export CUDA_VISIBLE_DEVICES=1
#
#for i in $(seq 13)
#do
#    python3 run_test_WL.py --index=$((i-1 + 65)) &
#    sleep 1
#done

