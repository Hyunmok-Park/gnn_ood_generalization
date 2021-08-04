#!/bin/sh

for i in $(seq 20)
do
  python3 -m dataset.gen_test_I2 --index=$((i-1)) &
done




