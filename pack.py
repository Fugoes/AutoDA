#!/usr/bin/env python3
import argparse
import os
import subprocess


def pack(dir, output):
    os.makedirs(output, exist_ok=True)
    basenames = ('CIFAR-10.h5', 'CIFAR-2', 'CIFAR-2-CLEAN.h5')
    cmds = ['rsync', '-av'] + [os.path.join(dir, basename) for basename in basenames] + [output]
    print('>>> ' + ' '.join(cmds))
    subprocess.run(cmds)

    cmds = ['ldconfig', '-p']
    print('>>> ' + ' '.join(cmds))
    r = subprocess.run(cmds, capture_output=True)

    cmds = ['rsync', '-av']
    for line in r.stdout.splitlines()[1:]:
        line = line.decode()
        words = [word.strip() for word in line.split(' ')]
        if words[0] == 'libtensorflow.so.2':
            cmds += [os.path.realpath(words[3])]
        elif words[0] == 'libtensorflow_framework.so.2':
            cmds += [os.path.realpath(words[3])]
    cmds += [output]
    print('>>> ' + ' '.join(cmds))
    subprocess.run(cmds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    pack(args.dir, args.output)
