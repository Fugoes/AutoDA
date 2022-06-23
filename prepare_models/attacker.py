#!/usr/bin/env python3
import argparse
import collections
import os
import sys

import filelock
import h5py
import numpy as np
import tensorflow as tf
import torch
import torchvision.models as models
import torchvision.datasets
import torchvision.transforms as transforms
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
from models import *

import Sign_OPT
import hsja

S0_INIT = 0.01


def remove_module_prefix_of_ckpt(ckpt):
    new_ckpt = dict()
    for key, value in ckpt.items():
        new_ckpt[key[7:]] = value
    return new_ckpt


def fn_predict(x):
    y = yield x
    return y


class FakeModel(object):
    def __init__(self):
        pass

    def predict_label(self, xs):
        label = yield xs[0]
        return [label]

    def predict_labels(self, xs):
        xs_len = len(xs)
        xs = xs.reshape((xs_len, -1))
        labels = []
        for i in range(len(xs)):
            label = yield xs[i]
            labels.append(label)
        return np.array(labels)


def load_cifar10():
    _, (xs_test, ys_test) = tf.keras.datasets.cifar10.load_data()
    xs_test = (xs_test / 255.0).astype(np.float32)
    xs_test = xs_test.reshape((len(xs_test), -1))
    return xs_test, ys_test.flatten()


backup_starting_points = dict()


def get_starting_point(x, y):
    count = 0
    while count < 100:
        count += 1
        noise = (x + np.random.rand() * np.random.randn(*x.shape)).clip(0.0, 1.0)
        if (yield from fn_predict(noise)) != y:
            return noise
    print("fallback to backup starting points")
    classes = list(backup_starting_points.keys())
    if y in classes: classes.remove(y)
    starting_point = backup_starting_points[np.random.choice(classes)]
    assert (yield from fn_predict(starting_point)) != y
    return starting_point


def autoda_generate(tac_program):
    lines = [line for line in tac_program.splitlines() if len(line) > 0]
    ret, _ = lines[-1].split('=')
    ret = ret.strip()
    program = ["def autoda_wrapper(s0, v0, v1, v2):"]
    for line in lines:
        program.append("    " + line)
    program.append("    return {}".format(ret))
    vs = {
        "np": np,
        "ADD": lambda a, b: a + b,
        "SUB": lambda a, b: a - b,
        "MUL": lambda a, b: a * b,
        "DIV": lambda a, b: a / b,
        "DOT": lambda a, b: np.dot(a, b),
        "NORM": lambda a: np.linalg.norm(a),
    }
    exec("\n".join(program), vs)
    return vs["autoda_wrapper"]


autoda_attack_0 = {
    "succ_rate": 0.25,
    "decay_factor": 0.95,
    "tune_lo": 0.5,
    "tune_hi": 1.5,
    "s0_hi_factor": 1.5,
    "program": autoda_generate("""
v3 = SUB(v0, v1)
s1 = NORM(v3)
v4 = DIV(v3, s1)
v5 = MUL(v2, s0)
v6 = MUL(v5, s1)
v4 = ADD(v5, v4)
s1 = DOT(v4, v5)
v3 = MUL(v3, s1)
v3 = ADD(v3, v1)
v3 = SUB(v3, v6)
""")
}

autoda_attack_1 = {
    "succ_rate": 0.25,
    "decay_factor": 0.95,
    "tune_lo": 0.5,
    "tune_hi": 1.5,
    "s0_hi_factor": 1.5,
    "program": autoda_generate("""
v3 = SUB(v0, v1)
s1 = NORM(v3)
v3 = DIV(v3, s1)
v4 = MUL(v2, s0)
v5 = ADD(v4, v3)
s2 = NORM(v1)
v5 = MUL(v5, s1)
s1 = DOT(v5, v4)
v4 = DIV(v3, s2)
v3 = ADD(v3, v4)
v3 = MUL(v3, s1)
v3 = ADD(v0, v3)
v3 = SUB(v3, v5)
""")
}


def autoda_attacker(autoda_attack, x, y, ss, idx):
    x_adv = yield from get_starting_point(x, y)
    program = autoda_attack["program"]

    s0_init = S0_INIT
    target_succ_rate = autoda_attack["succ_rate"]
    decay_factor = autoda_attack["decay_factor"]
    ratio_lo, ratio_hi = autoda_attack["tune_lo"], autoda_attack["tune_hi"]

    s0 = s0_init
    s0_hi = autoda_attack["s0_hi_factor"] * S0_INIT
    succ_rate = target_succ_rate

    failure_count, max_failure_count = 0, 5
    while True:
        v_out = program(s0, x, x_adv, np.random.normal(size=x.shape))
        noise = np.clip(v_out, 0.0, 1.0) - x
        noise_norm = np.linalg.norm(noise)
        dist = np.linalg.norm(x_adv - x)
        if noise_norm > dist:
            if failure_count < max_failure_count:
                failure_count += 1
                if dist > 1.0:
                    print("s0 = {} failed for {} times".format(s0, failure_count))
                continue
            else:
                if dist > 1.0:
                    print("!!! s0 = {} failed for {} times".format(s0, max_failure_count))
                failure_count = 0
            noise = (dist / noise_norm) * noise
        else:
            failure_count = 0
        candidate = x + noise
        is_adversarial = (yield from fn_predict(candidate.astype(np.float32))) != y

        succ_rate *= decay_factor
        if is_adversarial:
            succ_rate += 1 - decay_factor
        if succ_rate >= target_succ_rate:
            ratio = ((ratio_hi - 1.0) / (1.0 - target_succ_rate)) * (
                    succ_rate - target_succ_rate) + 1.0
        else:
            ratio = ((ratio_lo - 1.0) / (0.0 - target_succ_rate)) * (
                    succ_rate - target_succ_rate) + 1.0
        new_s0 = s0 * (ratio ** 0.1)
        if 1e-12 <= new_s0 <= s0_hi:
            s0 = new_s0

        if is_adversarial:
            x_adv = candidate
        ss[idx] = s0


def sign_opt_attacker(x, y):
    fake_model = FakeModel()
    sign_opt = Sign_OPT.OPT_attack_sign_SGD(fake_model, backup_starting_points)
    yield from sign_opt(np.array([x]), np.array([y]), targeted=False, query_limit=20001)


def hsja_attacker(x, y):
    fake_model = FakeModel()
    if x.shape[0] == 32 * 32 * 3:
        x = x.reshape(32, 32, 3)
    elif x.shape[0] == 299 * 299 * 3:
        x = x.reshape(299, 299, 3)
    elif x.shape[0] == 224 * 224 * 3:
        x = x.reshape(224, 224, 3)
    yield from hsja.hsja(fake_model, x, y, num_iterations=20001, backup_starting_points=backup_starting_points)


def hsja__attacker(x, y):
    fake_model = FakeModel()
    if x.shape[0] == 32 * 32 * 3:
        x = x.reshape(32, 32, 3)
    elif x.shape[0] == 299 * 299 * 3:
        x = x.reshape(299, 299, 3)
    elif x.shape[0] == 224 * 224 * 3:
        x = x.reshape(224, 224, 3)
    yield from hsja.hsja(fake_model, x, y, num_iterations=20001, backup_starting_points=backup_starting_points,
                         stepsize_search='grid_search')


def evolutionary_attacker(x, y):
    mu = 0.01
    sigma = 0.03
    decay_factor = 0.99
    c = 0.001

    x_adv = yield from get_starting_point(x, y)
    stats_adversarial = collections.deque(maxlen=30)
    pert_shape = x.shape
    N = np.prod(pert_shape)
    K = int(N / 20)

    evolution_path = np.zeros(pert_shape, dtype=np.float32)
    diagonal_covariance = np.ones(pert_shape, dtype=np.float32)

    while True:
        unnormalized_source_direction = x - x_adv
        source_norm = np.linalg.norm(unnormalized_source_direction)

        selection_probability = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
        selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

        perturbation = np.random.normal(0.0, 1.0, pert_shape).astype(np.float32)
        factor = np.zeros([N], dtype=np.float32)
        factor[selected_indices] = 1
        perturbation *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)

        perturbation_large = perturbation

        biased = x_adv + mu * unnormalized_source_direction
        candidate = biased + sigma * source_norm * perturbation_large / np.linalg.norm(
            perturbation_large)
        candidate = x - (x - candidate) / np.linalg.norm(x - candidate) * np.linalg.norm(x - biased)
        candidate = np.clip(candidate, 0.0, 1.0)
        candidate = candidate.astype(np.float32)

        candidate_label = yield from fn_predict(candidate)
        is_adversarial = candidate_label != y
        stats_adversarial.appendleft(is_adversarial)

        if is_adversarial:
            x_adv = candidate
            evolution_path = decay_factor * evolution_path + np.sqrt(
                1 - decay_factor ** 2) * perturbation
            diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)

        if len(stats_adversarial) == stats_adversarial.maxlen:
            p_step = np.mean(stats_adversarial)
            mu *= np.exp(p_step - 0.2)
            stats_adversarial.clear()


def boundary_attacker(x, y):
    spherical_step = 1e-2
    source_step = 1e-2
    step_adaptation = 1.5
    max_directions = 25

    x_min, x_max = 0.0, 1.0
    x_shape = x.shape
    x_dtype = np.float32

    def fn_is_adversarial(label):
        return label != y

    def fn_mean_square_distance(x1, x2):
        return np.mean((x1 - x2) ** 2) / ((x_max - x_min) ** 2)

    x_adv = yield from get_starting_point(x, y)
    dist = fn_mean_square_distance(x_adv, x)
    stats_spherical_adversarial = collections.deque(maxlen=100)
    stats_step_adversarial = collections.deque(maxlen=30)

    step, queries = 0, 0
    while True:
        step += 1

        unnormalized_source_direction = x - x_adv
        source_norm = np.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        do_spherical = (step % 10 == 0)

        for _ in range(max_directions):
            perturbation = np.random.normal(0.0, 1.0, x_shape).astype(x_dtype)
            dot = np.vdot(perturbation, source_direction)
            perturbation -= dot * source_direction
            perturbation *= spherical_step * source_norm / np.linalg.norm(perturbation)

            D = 1 / np.sqrt(spherical_step ** 2.0 + 1)
            direction = perturbation - unnormalized_source_direction
            spherical_candidate = np.clip(x + D * direction, x_min, x_max)

            new_source_direction = x - spherical_candidate
            new_source_direction_norm = np.linalg.norm(new_source_direction)
            length = source_step * source_norm

            deviation = new_source_direction_norm - source_norm
            length = max(0, length + deviation) / new_source_direction_norm
            candidate = np.clip(spherical_candidate + length * new_source_direction, x_min, x_max)

            if do_spherical:
                spherical_candidate_label = yield spherical_candidate
                spherical_is_adversarial = fn_is_adversarial(spherical_candidate_label)

                queries += 1
                stats_spherical_adversarial.appendleft(spherical_is_adversarial)

                if not spherical_is_adversarial:
                    continue

            candidate_label = yield candidate
            is_adversarial = fn_is_adversarial(candidate_label)

            queries += 1

            if do_spherical:
                stats_step_adversarial.appendleft(is_adversarial)

            if not is_adversarial:
                continue

            new_x_adv = candidate
            new_dist = fn_mean_square_distance(new_x_adv, x)
            break
        else:
            new_x_adv = None

        message = ''
        if new_x_adv is not None:
            abs_improvement = dist - new_dist
            rel_improvement = abs_improvement / dist
            message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100,
                                                              abs_improvement)
            x_adv_label, x_adv, dist = candidate_label, new_x_adv, new_dist

        if len(stats_step_adversarial) == stats_step_adversarial.maxlen and \
                len(stats_spherical_adversarial) == stats_spherical_adversarial.maxlen:
            p_spherical = np.mean(stats_spherical_adversarial)
            p_step = np.mean(stats_step_adversarial)
            n_spherical = len(stats_spherical_adversarial)
            n_step = len(stats_step_adversarial)

            if p_spherical > 0.5:
                message = 'Boundary too linear, increasing steps:'
                spherical_step *= step_adaptation
                source_step *= step_adaptation
            elif p_spherical < 0.2:
                message = 'Boundary too non-linear, decreasing steps:'
                spherical_step /= step_adaptation
                source_step /= step_adaptation
            else:
                message = None

            if message is not None:
                stats_spherical_adversarial.clear()

            if p_step > 0.5:
                message = 'Success rate too high, increasing source step:'
                source_step *= step_adaptation
            elif p_step < 0.2:
                message = 'Success rate too low, decreasing source step:'
                source_step /= step_adaptation
            else:
                message = None

            if message is not None:
                stats_step_adversarial.clear()


def main(args):
    global S0_INIT
    if args.model == "inception_v3" or args.model == "wrn":
        S0_INIT = 0.001
        xs_test = np.load("ImageNet.npy")
        xs_backup = xs_test[:100].astype(np.float32) / 255.0
        xs_test = xs_test[args.offset:args.offset + args.count].astype(np.float32) / 255.0
        xs_test = xs_test.reshape((len(xs_test), -1))
        with open("val.txt") as f:
            ys_test = []
            for line in f:
                _, y = line.split(' ')
                ys_test.append(int(y))
                if len(ys_test) == 1000:
                    break
        ys_test = np.array(ys_test)[args.offset:args.offset + args.count]
        ids = np.arange(args.offset, args.offset + args.count)
        eps_succ = np.sqrt(299 * 299 * 3 * 0.001)
    elif args.model == "resnet101" or args.model == "wide_resnet50_2" or args.model == "resnet50":
        S0_INIT = 0.001
        xs_test = np.load("ImageNet224.npy")
        xs_backup = xs_test[:100].astype(np.float32) / 255.0
        xs_test = xs_test[args.offset:args.offset + args.count].astype(np.float32) / 255.0
        xs_test = xs_test.reshape((len(xs_test), -1))
        with open("val.txt") as f:
            ys_test = []
            for line in f:
                _, y = line.split(' ')
                ys_test.append(int(y))
                if len(ys_test) == 1000:
                    break
        ys_test = np.array(ys_test)[args.offset:args.offset + args.count]
        ids = np.arange(args.offset, args.offset + args.count)
        eps_succ = np.sqrt(224 * 224 * 3 * 0.001)
    else:
        xs_test, ys_test = load_cifar10()
        xs_backup = xs_test[:100]
        xs_test = xs_test[args.offset:args.offset + args.count]
        ys_test = ys_test[args.offset:args.offset + args.count]
        ids = np.arange(args.offset, args.offset + args.count)
        eps_succ = 1

    if args.model == "CIFAR-2_0_1":
        loaded = tf.saved_model.load("CIFAR-2_0_1/")
        batch_size = 1000

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                return tf.cast(loaded.logits(xs_batch) > 0.0, tf.int32).numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    ys_batch.append(tf.cast(loaded.logits(xs) > 0.0, tf.int32).numpy())
                return np.concatenate(ys_batch)

        conds = ys_test == 0
        xs_test, ids = xs_test[conds], ids[conds]
        conds = run_model(xs_test) == 0
        xs_test, ids = xs_test[conds], ids[conds]
        ys_test = np.zeros(len(xs_test), dtype=np.int32)
    elif args.model == "Madry":
        loaded = tf.saved_model.load("Madry")
        batch_size = 40

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                return loaded.run(xs_batch)["labels"].numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    ys_batch.append(loaded.run(xs)["labels"].numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "Robustness_nat":
        loaded, _ = make_and_restore_model(arch='resnet50', dataset=CIFAR(),
                                           resume_path='cifar_nat.pt')
        loaded.eval()
        batch_size = 100

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_batch_torch = torch.tensor(xs_batch.reshape((xs_batch_len, 32, 32, 3)),
                                              requires_grad=False)
                xs_batch_torch = xs_batch_torch.transpose(1, 2).transpose(1, 3).contiguous()
                return loaded(xs_batch_torch.cuda())[0].argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.tensor(xs.reshape((len(xs), 32, 32, 3)), requires_grad=False)
                    xs_torch = xs_torch.transpose(1, 2).transpose(1, 3).contiguous()
                    ys_batch.append(loaded(xs_torch.cuda())[0].argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "Robustness_l2_1_0":
        loaded, _ = make_and_restore_model(arch='resnet50', dataset=CIFAR(),
                                           resume_path='cifar_l2_1_0.pt')
        loaded.eval()
        batch_size = 100

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_batch_torch = torch.tensor(xs_batch.reshape((xs_batch_len, 32, 32, 3)),
                                              requires_grad=False)
                xs_batch_torch = xs_batch_torch.transpose(1, 2).transpose(1, 3).contiguous()
                return loaded(xs_batch_torch.cuda())[0].argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.tensor(xs.reshape((len(xs), 32, 32, 3)), requires_grad=False)
                    xs_torch = xs_torch.transpose(1, 2).transpose(1, 3).contiguous()
                    ys_batch.append(loaded(xs_torch.cuda())[0].argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "densenet":
        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        loaded = DenseNet121()
        loaded.eval()
        loaded.cuda()

        ckpt = torch.load("./checkpoint/densenet-ckpt.pth")
        loaded.load_state_dict(remove_module_prefix_of_ckpt(ckpt["net"]))
        print(ckpt["acc"], ckpt["epoch"])

        batch_size = 50

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_batch_torch = torch.tensor(xs_batch.reshape((xs_batch_len, 32, 32, 3)),
                                              requires_grad=False)
                xs_batch_torch = xs_batch_torch.transpose(1, 2).transpose(1, 3).contiguous()
                xs_batch_torch = transform_test(xs_batch_torch.cuda())
                return loaded(xs_batch_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.tensor(xs.reshape((len(xs), 32, 32, 3)), requires_grad=False)
                    xs_torch = xs_torch.transpose(1, 2).transpose(1, 3).contiguous()
                    xs_torch = transform_test(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "dla":
        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        loaded = SimpleDLA()
        loaded.eval()
        loaded.cuda()

        ckpt = torch.load("./checkpoint/dla-ckpt.pth")
        loaded.load_state_dict(remove_module_prefix_of_ckpt(ckpt["net"]))
        print(ckpt["acc"], ckpt["epoch"])

        batch_size = 100

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_batch_torch = torch.tensor(xs_batch.reshape((xs_batch_len, 32, 32, 3)),
                                              requires_grad=False)
                xs_batch_torch = xs_batch_torch.transpose(1, 2).transpose(1, 3).contiguous()
                xs_batch_torch = transform_test(xs_batch_torch.cuda())
                return loaded(xs_batch_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.tensor(xs.reshape((len(xs), 32, 32, 3)), requires_grad=False)
                    xs_torch = xs_torch.transpose(1, 2).transpose(1, 3).contiguous()
                    xs_torch = transform_test(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "dpn":
        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        loaded = DPN92()
        loaded.eval()
        loaded.cuda()

        ckpt = torch.load("./checkpoint/dpn-ckpt.pth")
        loaded.load_state_dict(remove_module_prefix_of_ckpt(ckpt["net"]))
        print(ckpt["acc"], ckpt["epoch"])

        batch_size = 50

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_batch_torch = torch.tensor(xs_batch.reshape((xs_batch_len, 32, 32, 3)),
                                              requires_grad=False)
                xs_batch_torch = xs_batch_torch.transpose(1, 2).transpose(1, 3).contiguous()
                xs_batch_torch = transform_test(xs_batch_torch.cuda())
                return loaded(xs_batch_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.tensor(xs.reshape((len(xs), 32, 32, 3)), requires_grad=False)
                    xs_torch = xs_torch.transpose(1, 2).transpose(1, 3).contiguous()
                    xs_torch = transform_test(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "resnet50":
        loaded = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval()
        loaded.cuda()
        batch_size = 25
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_torch = torch.as_tensor(xs_batch.reshape((xs_batch_len, 3, 224, 224))).contiguous()
                xs_torch = normalize(xs_torch.cuda())
                return loaded(xs_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.as_tensor(xs.reshape((len(xs), 3, 224, 224))).contiguous()
                    xs_torch = normalize(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "resnet101":
        loaded = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True).eval()
        loaded.cuda()
        batch_size = 25
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_torch = torch.as_tensor(xs_batch.reshape((xs_batch_len, 3, 224, 224))).contiguous()
                xs_torch = normalize(xs_torch.cuda())
                return loaded(xs_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.as_tensor(xs.reshape((len(xs), 3, 224, 224))).contiguous()
                    xs_torch = normalize(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "wide_resnet50_2":
        loaded = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).eval()
        loaded.cuda()
        batch_size = 25
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_torch = torch.as_tensor(xs_batch.reshape((xs_batch_len, 3, 224, 224))).contiguous()
                xs_torch = normalize(xs_torch.cuda())
                return loaded(xs_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.as_tensor(xs.reshape((len(xs), 3, 224, 224))).contiguous()
                    xs_torch = normalize(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "inception_v3":
        normalize = torch.jit.script(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
        loaded = models.inception_v3(pretrained=True)
        loaded.eval()
        loaded.cuda()

        batch_size = 45

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_torch = torch.as_tensor(
                    xs_batch.reshape((xs_batch_len, 3, 299, 299))).contiguous()
                xs_torch = normalize(xs_torch.cuda())
                return loaded(xs_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.as_tensor(xs.reshape((len(xs), 3, 299, 299))).contiguous()
                    xs_torch = normalize(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    elif args.model == "wrn":
        normalize = torch.jit.script(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
        loaded = models.wide_resnet50_2(pretrained=True)
        loaded.eval()
        loaded.cuda()

        batch_size = 20

        def run_model(xs_batch):
            xs_batch_len = len(xs_batch)
            if xs_batch_len <= batch_size:
                xs_torch = torch.as_tensor(
                    xs_batch.reshape((xs_batch_len, 3, 299, 299))).contiguous()
                xs_torch = normalize(xs_torch.cuda())
                return loaded(xs_torch).argmax(axis=1).cpu().numpy()
            else:
                batches = xs_batch_len // batch_size
                if xs_batch_len % batch_size != 0: batches += 1
                ys_batch = []
                for xs in np.array_split(xs_batch, batches):
                    xs_torch = torch.as_tensor(xs.reshape((len(xs), 3, 299, 299))).contiguous()
                    xs_torch = normalize(xs_torch.cuda())
                    ys_batch.append(loaded(xs_torch).argmax(axis=1).cpu().numpy())
                return np.concatenate(ys_batch)

        conds = run_model(xs_test) == ys_test
        xs_test, ys_test, ids = xs_test[conds], ys_test[conds], ids[conds]
    else:
        raise NotImplementedError

    lock = filelock.FileLock(args.output + ".lock")
    with lock.acquire():
        if os.path.isfile(args.output):
            with h5py.File(args.output, "r") as f:
                for idx, img_id in enumerate(ids):
                    if "{}".format(img_id) in f.keys():
                        print("exit")
                        sys.exit(0)

    xs_backup_labels = run_model(xs_backup)
    for x_backup_label, x_backup in zip(xs_backup_labels, xs_backup):
        backup_starting_points[int(x_backup_label)] = x_backup.flatten()

    attackers = []
    xs_run = np.zeros_like(xs_test, dtype=np.float32)

    if args.method.startswith("autoda_"):
        _, autoda_id = args.method.split("_")
        autoda_attack = globals()["autoda_attack_" + autoda_id]
        ss = np.zeros(len(xs_run))
        for idx, (x, y) in enumerate(zip(xs_test, ys_test)):
            attackers.append(autoda_attacker(autoda_attack, x, y, ss, idx))
    elif args.method == "sign_opt":
        for x, y in zip(xs_test, ys_test):
            attackers.append(sign_opt_attacker(x, y))
    elif args.method == "hsja":
        for x, y in zip(xs_test, ys_test):
            attackers.append(hsja_attacker(x, y))
    elif args.method == "hsja_":
        for x, y in zip(xs_test, ys_test):
            attackers.append(hsja__attacker(x, y))
    elif args.method == "evolutionary":
        for x, y in zip(xs_test, ys_test):
            attackers.append(evolutionary_attacker(x, y))
    elif args.method == "boundary":
        for x, y in zip(xs_test, ys_test):
            attackers.append(boundary_attacker(x, y))
    else:
        raise NotImplementedError

    for idx, attacker in enumerate(attackers):
        try:
            xs_run[idx] = next(attacker)
        except StopIteration:
            pass
    xs_run = xs_run.clip(0.0, 1.0)

    xs_adv = (xs_test + 10).astype(np.float32)
    history = np.zeros((20000, len(xs_adv)), dtype=np.float32)
    best_dists = np.array([np.inf for _ in range(len(xs_test))], dtype=np.float32)

    trajectory_interval = args.trajectory_interval

    if trajectory_interval > 0:
        print("saving trajectory")
        lock = filelock.FileLock("trajectory_" + args.output + ".lock")
        with lock.acquire():
            with h5py.File("trajectory_" + args.output, "a") as f:
                f.create_dataset("xs_test", data=xs_test)
                f.create_dataset("ys_test", data=ys_test)

    for step in range(20000):
        # this step
        ys_run = run_model(xs_run)
        xs_run_dists = np.linalg.norm(xs_run - xs_test, axis=1)
        to_update = np.logical_and(xs_run_dists <= best_dists, ys_run != ys_test)
        best_dists[to_update] = xs_run_dists[to_update]
        xs_adv[to_update] = xs_run[to_update]
        history[step] = best_dists
        if args.method.startswith("autoda_"):
            print("step={:5d}  dists={:8.6f},{:8.6f}  succ={:.3f}  count={:4d}  ss={},{},{}".format(
                step, best_dists.mean(), np.median(best_dists), (best_dists < eps_succ).mean(), (ys_run != ys_test).sum(),
                ss.min(), ss.mean(), ss.max(),
            ))
        else:
            print("step={:5d}  dists={:8.6f},{:8.6f}  succ={:.3f}  count={:4d}".format(
                step, best_dists.mean(), np.median(best_dists), (best_dists < eps_succ).mean(), (ys_run != ys_test).sum(),
            ))
        if trajectory_interval > 0 and (step + 1) in (100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000):
            print("saving trajectory")
            lock = filelock.FileLock("trajectory_" + args.output + ".lock")
            with lock.acquire():
                with h5py.File("trajectory_" + args.output, "a") as f:
                    f.create_dataset("{}".format(step + 1), data=xs_adv)
        # next step
        for idx, attacker in enumerate(attackers):
            try:
                xs_run[idx] = attacker.send(ys_run[idx])
            except StopIteration:
                pass
        xs_run = xs_run.clip(0.0, 1.0)

    xs_adv_labels = run_model(xs_adv)
    print(xs_adv_labels != ys_test, (xs_adv_labels != ys_test).mean())

    lock = filelock.FileLock(args.output + ".lock")
    with lock.acquire():
        with h5py.File(args.output, "a") as f:
            for idx, img_id in enumerate(ids):
                f.create_dataset("{}".format(img_id), data=history[:, idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--model", type=str, choices=[
        "CIFAR-2_0_1", "Madry", "Robustness_nat", "Robustness_l2_1_0", "inception_v3", "densenet", "dla", "dpn", "wrn",
        "resnet101", "wide_resnet50_2", "resnet50",
    ], required=True)
    parser.add_argument("--label", type=int, required=False)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--trajectory-interval", type=int, default=0)
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    main(args)