#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf


def generate_dataset():
    (xs_train, ys_train), (xs_test, ys_test) = tf.keras.datasets.cifar10.load_data()
    ys_train, ys_test = ys_train.reshape(-1), ys_test.reshape(-1)

    xs_shape = (-1, 32 * 32 * 3)
    with h5py.File('CIFAR-10.h5', 'w') as f:
        for i in range(10):
            f.create_dataset('test_{}'.format(i), data=np.reshape(xs_test[np.where(ys_test == i)], xs_shape))
        for i in range(10):
            f.create_dataset('train_{}'.format(i), data=np.reshape(xs_train[np.where(ys_train == i)], xs_shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    generate_dataset()
