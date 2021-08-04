#!/bin/sh

for i in $(seq 16)
do
#  python3 WL_flex_II_sh.py --index=$((i-1)) &

  python3 WL_flex_II_sh_bimodal.py --index=$((i-1)) &
done




