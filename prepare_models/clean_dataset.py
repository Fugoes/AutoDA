#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf


def load_dataset(class_0):
    with h5py.File('CIFAR-10.h5', 'r') as f:
        xs_train = np.array(f.get('train_{}'.format(class_0)))
        ys_train = np.zeros(len(xs_train), dtype=np.uint8)
        xs_test = np.array(f.get('test_{}'.format(class_0)))
        ys_test = np.zeros(len(xs_test), dtype=np.uint8)
        xs_train, xs_test = xs_train.astype(np.float32) / 255.0, xs_test.astype(np.float32) / 255.0
    return (xs_train, ys_train), (xs_test, ys_test)


def generate_clean_dataset():
    loaded = tf.saved_model.load('CIFAR-2')
    with h5py.File('CIFAR-2-CLEAN.h5', 'w') as f:
        for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
            _, (xs_test_0, _) = load_dataset(i)
            _, (xs_test_1, _) = load_dataset(j)
            run_model = getattr(loaded, 'run_{}_{}'.format(i, j))
            idx_0 = np.arange(len(xs_test_0))[np.logical_not(run_model(xs_test_0)['labels'].numpy())].astype(np.int32)
            idx_1 = np.arange(len(xs_test_1))[run_model(xs_test_1)['labels'].numpy()].astype(np.int32)
            f.create_dataset('model_{}_{}_0'.format(i, j), data=idx_0)
            f.create_dataset('model_{}_{}_1'.format(i, j), data=idx_1)
            print('{},{} {} {}'.format(i, j, len(idx_0), len(idx_1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    generate_clean_dataset()
