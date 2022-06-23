#!/bin/bash
for i in {0..9}; do
    for j in {0..9}; do
        if [ $i -ne $j ]; then
            python3 -u train.py --dir $1 --class-0 $i --class-1 $j --epochs 100
        fi
    done
done
