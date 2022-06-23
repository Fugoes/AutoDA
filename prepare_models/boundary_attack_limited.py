#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
import collections
import time


def load_dataset(class_0):
    with h5py.File('CIFAR-10.h5', 'r') as f:
        xs_train = np.array(f.get('train_{}'.format(class_0)))
        ys_train = np.zeros(len(xs_train), dtype=np.uint8)
        xs_test = np.array(f.get('test_{}'.format(class_0)))
        ys_test = np.zeros(len(xs_test), dtype=np.uint8)
        xs_train, xs_test = xs_train.astype(np.float32) / 255.0, xs_test.astype(np.float32) / 255.0
    return (xs_train, ys_train), (xs_test, ys_test)


def boundary_attack(xs_adv, idx, x, x_adv, x_shape, x_min, x_max,
                    spherical_step, src_steps, success):
    x_label = yield x
    if x_label == 1:
        xs_adv[idx] = x
        return

    xs_adv[idx] = x_adv

    step = 0
    succ_rate = 0.1
    n = 10
    while True:
        step += 1

        # attack
        src_direction = x - x_adv
        src_direction_norm = np.linalg.norm(src_direction)
        src_direction_unit = src_direction / src_direction_norm

        pert = np.random.normal(0.0, 1.0, x_shape).astype(np.float32)
        pert -= np.vdot(pert, src_direction_unit) * src_direction_unit
        pert *= spherical_step * src_direction_norm / np.linalg.norm(pert)

        factor = np.sqrt(spherical_step ** 2.0 + 1)
        new_src_direction = (src_direction - pert) / factor
        new_src_direction_norm = np.linalg.norm(new_src_direction)
        spherical_candidate = x - new_src_direction
        length = src_steps[idx] * src_direction_norm

        deviation = new_src_direction_norm - src_direction_norm
        length = max(0, length + deviation) / new_src_direction_norm
        candidate = np.clip(spherical_candidate + length * new_src_direction, x_min, x_max)
        # end attack

        candidate_label = yield candidate

        # learn
        succ_rate *= 1 - 1 / n
        succ_rate += candidate_label / n
        src_steps[idx] *= (succ_rate - 0.2) / n + 1.0
        # end learn

        if candidate_label == 1:
            x_adv = candidate
            xs_adv[idx] = candidate
            success.append(idx)


def attack(class_0, class_1, output, random_starting_points):
    loaded = tf.saved_model.load('CIFAR-2')
    _, (xs_test, _) = load_dataset(class_0)
    with h5py.File('CIFAR-2-CLEAN.h5', 'r') as f:
        xs_test = xs_test[f.get('model_{}_{}_0'.format(class_0, class_1))]

    run_model = getattr(loaded, 'run_{}_{}'.format(class_0, class_1))

    if random_starting_points:
        print("Load random starting points")
        xs_1 = np.random.rand(0, 32 * 32 * 3).astype(np.float)
        while len(xs_1) < len(xs_test):
            x_try = np.random.rand(1000, 32 * 32 * 3)
            labels = run_model(x_try)['labels']
            xs_1 = np.concatenate([xs_1, x_try[labels]])
        xs_1 = xs_1[:len(xs_test)]
    else:
        print("Load CIFAR-2-CLEAN")
        _, (xs_test_1, _) = load_dataset(class_1)
        with h5py.File('CIFAR-2-CLEAN.h5', 'r') as f:
            xs_1 = xs_test_1[np.random.choice(f.get('model_{}_{}_1'.format(class_0, class_1)), len(xs_test))]

    max_queries = 20000
    spherical_step = 2e-2
    src_step = 1e-2

    success = []
    xs_adv = xs_1.copy().astype(np.float)
    xs = np.zeros_like(xs_test, dtype=np.float)
    src_steps = np.zeros(len(xs_test), dtype=np.float) + src_step
    gs = [boundary_attack(xs_adv, i, xs_test[i], xs_1[i], (3 * 32 * 32,), 0.0, 1.0,
                          spherical_step, src_steps, success)
          for i in range(len(xs_test))]
    orig_dists = np.linalg.norm(xs_test - xs_1, axis=1)

    for idx, g in enumerate(gs):
        try:
            xs[idx] = next(g)
        except StopIteration:
            pass

    with h5py.File(output, 'w') as f:
        f.clear()
    for step in range(max_queries):
        dists = np.linalg.norm(xs_adv - xs_test, axis=1)
        with h5py.File(output, 'r+') as f:
            f.create_dataset('{}'.format(step), data=dists)
        start = time.time()
        labels = run_model(xs)['labels'].numpy().astype(np.int)
        end = time.time()
        gpu_time = end - start
        success.clear()
        start = time.time()
        for idx, g in enumerate(gs):
            xs[idx] = g.send(labels[idx])
        end = time.time()
        cpu_time = end - start
        print(
            'step={:5d}  dist_mean={:10.6f}  ratio_mean={:10.6f}  src_step={:10.6f},{:10.6f},{:10.6f}  succ={:.3f}  '
            'count={:4d}  cpu_time={:.2f}ms  gpu_time={:.2f}ms'.format(
                step + 1, dists.mean(), (dists / orig_dists).mean(), src_steps.min(), src_steps.mean(), src_steps.max(),
                np.mean(dists < np.sqrt(0.001 * 32 * 32 * 3)), len(success),
                cpu_time * 1000, gpu_time * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--class-0', type=int, required=True)
    parser.add_argument('--class-1', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--random-starting-points', action='store_true')
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    attack(args.class_0, args.class_1, args.output, args.random_starting_points)
