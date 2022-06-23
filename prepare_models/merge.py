#!/usr/bin/env python3
import os
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

wrapper_header = '''
class Wrapper(tf.Module):
    def __init__(self, models):
        super().__init__()'''
wrapper_init_template = '''
        self.model_{0}_{1} = models[{0}][{1}]'''
wrapper_run_template = '''

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 32 * 32 * 3], dtype=tf.float32, name='xs')])
    def run_{0}_{1}(self, xs):
        logits = self.model_{0}_{1}(xs)[0]
        probabilities = tf.math.sigmoid(logits)
        labels = logits > 0.0
        return {{'logits': logits, 'probabilities': probabilities, 'labels': labels}}'''


def merge():
    models = {}
    for i in range(10):
        models[i] = {}
        for j in range(10):
            if i != j:
                model = tf.saved_model.load('CIFAR-2_{}_{}'.format(i, j))
                frozen_model = tf.function(lambda xs: model.logits(xs)) \
                    .get_concrete_function(tf.TensorSpec(shape=[None, 32 * 32 * 3], dtype=tf.float32, name='xs'))
                frozen_model = convert_variables_to_constants_v2(frozen_model)
                models[i][j] = frozen_model

    wrapper_src = wrapper_header
    for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
        wrapper_src += wrapper_init_template.format(i, j)
    for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
        wrapper_src += wrapper_run_template.format(i, j)
    exec(wrapper_src)
    wrapper = locals()['Wrapper'](models)
    signatures = {}
    for i, j in [(i, j) for i in range(10) for j in range(10) if i != j]:
        signatures['run_{}_{}'.format(i, j)] = getattr(wrapper, 'run_{}_{}'.format(i, j)) \
            .get_concrete_function(tf.TensorSpec(shape=[None, 32 * 32 * 3], dtype=tf.float32, name='xs'))
    tf.saved_model.save(wrapper, 'CIFAR-2', signatures=signatures)
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    merge()
