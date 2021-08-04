#!/bin/sh
#export CUDA_VISIBLE_DEVICES=0
#python3 -m dataset.gen_test_II2


#python3 -m Find_uni_graph2 --kk_index=0 &
#python3 -m Find_uni_graph2 --kk_index=1 &
#python3 -m Find_uni_graph --kk_index=2 &
#python3 -m Find_uni_graph --kk_index=3 &
#python3 -m Find_uni_graph --kk_index=4 &
#python3 -m Find_uni_graph --kk_index=5 &
#python3 -m Find_uni_graph --kk_index=6 &
#python3 -m Find_uni_graph --kk_index=7 &
#python3 -m Find_uni_graph --kk_index=8 &
#python3 -m Find_uni_graph --kk_index=9 &
#python3 -m Find_uni_graph --kk_index=10 &
#python3 -m Find_uni_graph --kk_index=11 &
#python3 -m Find_uni_graph --kk_index=12 &


#python3 -m Find_uni_graph_100 --index=0 &
#python3 -m Find_uni_graph_100 --index=1 &
#python3 -m Find_uni_graph_100 --index=2 &
#python3 -m Find_uni_graph_100 --index=3 &
#python3 -m Find_uni_graph_100 --index=4 &

#for i in $(seq 10)
#do
#  python3 -m Find_uni_graph_2_100 --index=4 --sub_index=$((i-1)) &
#done

python3 -m Find_uni_graph --index=0 &
python3 -m Find_uni_graph --index=1 &
python3 -m Find_uni_graph --index=2 &
python3 -m Find_uni_graph --index=3 &
