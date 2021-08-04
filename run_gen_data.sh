#!/bin/sh
#export CUDA_VISIBLE_DEVICES=0
#python3 -m dataset.gen_test_II2

for i in {1..10}
do
    python3 -m dataset.gen_train --sample_id=$i > "../data_christmas/out_$i.txt" &
    #python3 -m dataset.gen_val --sample_id=$i > "../data_christmas/out_$i.txt" &
    #python3 -m dataset.gen_test --sample_id=$i > "../data_christmas/out_$i.txt" &
    #python3 run_exp_local.py -c config/node_gnn_$i.yaml &
done

#  run.sh
#
#
#  Created by KiJung on 7/13/17.
#
#  https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
