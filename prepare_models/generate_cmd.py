#!/usr/bin/env python3
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--models', type=str, required=True)
    parser.add_argument('--methods', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--offset', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()

    models = args.models.split(',')
    methods = args.methods.split(',')
    for model in models:
        for method in methods:
            for lo in range(args.offset, args.offset + args.count, args.batch_size):
                hi = min(lo + args.batch_size, args.offset + args.count)
                print("python3 -u attacker.py "
                      "--dir {} "
                      "--method {} "
                      "--model {} "
                      "--offset {} "
                      "--count {} "
                      "--output {}".format(
                    args.dir, method, model, lo, hi - lo,
                    os.path.join(args.output, "{}_{}.h5".format(method, model))
                ))
