#!/usr/bin/env python3
import os
import argparse
from tensorflow.python.tools import saved_model_utils

src_head = '''#ifndef AUTODA_MODEL_INFO_HPP
#define AUTODA_MODEL_INFO_HPP

namespace {

const std::tuple<unsigned, unsigned, const char *, const char *> MODEL_ENTRIES[] = {'''

src_entry = '''
    {{{0}, {1}, "{2}", "{3}"}},'''

src_tail = '''
};

}

#endif //AUTODA_MODEL_INFO_HPP'''


def generate_header():
    src = src_head
    sig = saved_model_utils.get_meta_graph_def('CIFAR-2', 'serve').signature_def
    for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
        input_name, _ = sig['run_{}_{}'.format(i, j)].inputs['xs'].name.split(':')
        output_name, _ = sig['run_{}_{}'.format(i, j)].outputs['labels'].name.split(':')
        assert sig['run_{}_{}'.format(i, j)].outputs['labels'].name == '{}:{}'.format(output_name, 0)
        assert sig['run_{}_{}'.format(i, j)].outputs['logits'].name == '{}:{}'.format(output_name, 1)
        assert sig['run_{}_{}'.format(i, j)].outputs['probabilities'].name == '{}:{}'.format(output_name, 2)
        src += src_entry.format(i, j, input_name, output_name)
    src += src_tail
    return src


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    src = generate_header()
    print(src)
