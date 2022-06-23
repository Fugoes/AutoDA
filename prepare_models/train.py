#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
import efficientnet.model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def load_dataset(class_0, class_1):
    xs_shape = (-1, 32 * 32 * 3)
    with h5py.File('CIFAR-10.h5', 'r') as f:
        xs_train_0 = np.array(f.get('train_{}'.format(class_0)))
        xs_train_0 = np.reshape(xs_train_0, xs_shape)
        xs_train_1 = np.array(f.get('train_{}'.format(class_1)))
        xs_train_1 = np.reshape(xs_train_1, xs_shape)
        xs_train = np.concatenate([xs_train_0, xs_train_1])
        ys_train = np.concatenate([np.zeros(len(xs_train_0), dtype=np.uint8), np.ones(len(xs_train_1), dtype=np.uint8)])
        xs_test_0 = np.array(f.get('test_{}'.format(class_0)))
        xs_test_0 = np.reshape(xs_test_0, xs_shape)
        xs_test_1 = np.array(f.get('test_{}'.format(class_1)))
        xs_test_1 = np.reshape(xs_test_1, xs_shape)
        xs_test = np.concatenate([xs_test_0, xs_test_1])
        ys_test = np.concatenate([np.zeros(len(xs_test_0), dtype=np.uint8), np.ones(len(xs_test_1), dtype=np.uint8)])
        xs_train, xs_test = xs_train.astype(np.float32) / 255.0, xs_test.astype(np.float32) / 255.0
    return (xs_train, ys_train), (xs_test, ys_test)


def load_model():
    xs = tf.keras.Input(shape=32 * 32 * 3, dtype=tf.float32, name='xs')
    xs_input = tf.keras.layers.Reshape((32, 32, 3))(xs)
    efn_base = efficientnet.model.EfficientNet(
        width_coefficient=0.5,
        depth_coefficient=0.5,
        default_resolution=1000000,
        dropout_rate=0.1,
        drop_connect_rate=0.2,
        depth_divisor=8,
        weights='./efn_base.h5',
        input_tensor=xs_input,
        input_shape=(32, 32, 3),
        include_top=False,
        backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils,
    )
    model = tf.keras.Sequential(name='efn_cifar2')
    model.add(efn_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'))
    model.add(tf.keras.layers.Dropout(0.2, name='dropout'))
    model.add(tf.keras.layers.Dense(1, name='logits'))
    return efn_base, model


def train(class_0, class_1, epochs):
    (xs_train, ys_train), (xs_test, ys_test) = load_dataset(class_0, class_1)
    efn_base, model = load_model()
    ckpt_path = 'CIFAR-2_{}_{}'.format(class_0, class_1)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # fine-tune
    efn_base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    model.fit(xs_train, ys_train, epochs=5, batch_size=200, validation_data=(xs_test, ys_test))
    # retrain all parameters
    efn_base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=loss, metrics=['accuracy'])
    cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path + '.h5', save_weights_only=True, monitor='val_accuracy',
                                            mode='max', save_best_only=True, verbose=1)
    model.fit(xs_train, ys_train, epochs=epochs, batch_size=200, validation_data=(xs_test, ys_test), callbacks=[cb])
    # load best checkpoint
    model.load_weights(ckpt_path + '.h5')
    os.remove(ckpt_path + '.h5')
    # convert to constant graph
    xs_spec = tf.TensorSpec(shape=(None, 32 * 32 * 3), dtype=tf.float32, name='xs')
    frozen_model = tf.function(lambda xs: model(xs, training=False)).get_concrete_function(xs_spec)
    frozen_model = convert_variables_to_constants_v2(frozen_model)

    # wrapper class for tf.saved_model.save
    class Wrapper(tf.Module):
        def __init__(self):
            super().__init__()
            self.model = frozen_model

        @tf.function(input_signature=[xs_spec])
        def logits(self, xs):
            return tf.reshape(self.model(xs)[0], [-1])

    # save the model as tf2's SavedModel
    wrapper = Wrapper()
    tf.saved_model.save(wrapper, ckpt_path, signatures={'logits': wrapper.logits.get_concrete_function(xs_spec)})
    # load the model and run tests
    tf.keras.backend.clear_session()
    loaded = tf.saved_model.load(ckpt_path)
    acc = np.mean((loaded.logits(xs_test).numpy() > 0).astype(np.int) == ys_test)
    with open('CIFAR-2.txt', 'a') as f:
        f.write('{} {}\n'.format(ckpt_path, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--class-0', type=int, required=True)
    parser.add_argument('--class-1', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    train(args.class_0, args.class_1, args.epochs)
