#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf


def load_dataset(class_id):
    with h5py.File('CIFAR-10.h5', 'r') as f:
        return np.array(f.get('test_{}'.format(class_id))).astype(np.float32) / 255.0


def test():
    ds = [load_dataset(i) for i in range(10)]
    model = tf.saved_model.load('CIFAR-2')
    accs = []
    for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
        rs_0 = getattr(model, 'run_{}_{}'.format(i, j))(ds[i])
        acc_0 = np.mean(rs_0['labels'].numpy() == 0)
        rs_1 = getattr(model, 'run_{}_{}'.format(i, j))(ds[j])
        acc_1 = np.mean(rs_1['labels'].numpy() == 1)
        print(i, acc_0, j, acc_1, (acc_0 + acc_1) / 2)
        accs.append(acc_0)
        accs.append(acc_1)
    print(np.mean(accs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    test()
