#!/bin/sh

for i in $(seq 16)
do
  python3 -m dataset.WL_flex_II_sh --index=$((i-1)) &
done


#for i in 19
#do
#  python3 -m dataset.WL_flex_II_sh --index=$((i)) &
#done


