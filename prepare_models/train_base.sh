#!/bin/bash
./train_base.py --dir $1 --epochs 100 --learning-rate 1e-2
./train_base.py --dir $1 --epochs 100 --learning-rate 1e-3
./train_base.py --dir $1 --epochs 100 --learning-rate 1e-4
