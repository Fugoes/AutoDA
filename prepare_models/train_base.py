#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import efficientnet.model


def base(epochs, learning_rate):
    (xs_train, ys_train), (xs_test, ys_test) = tf.keras.datasets.cifar10.load_data()
    xs_train, xs_test = xs_train.astype(np.float32) / 255.0, xs_test.astype(np.float32) / 255.0
    ys_train, ys_test = np.reshape(ys_train, (-1,)), np.reshape(ys_test, (-1,))

    xs = tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32, name='xs')
    efn_base = efficientnet.model.EfficientNet(
        width_coefficient=0.5,
        depth_coefficient=0.5,
        default_resolution=1000000,
        dropout_rate=0.1,
        drop_connect_rate=0.2,
        depth_divisor=8,
        weights=None,
        input_tensor=xs,
        input_shape=(32, 32, 3),
        include_top=False,
        backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils,
    )
    model = tf.keras.Sequential(name='efn_cifar10')
    model.add(efn_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'))
    model.add(tf.keras.layers.Dropout(0.2, name='top_dropout'))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='probs'))
    model.summary()

    if os.path.exists('base.h5'):
        model.load_weights('base.h5')

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    cb = tf.keras.callbacks.ModelCheckpoint(filepath='base.h5', save_weights_only=True, monitor='val_accuracy',
                                            mode='max', save_best_only=True, verbose=1)
    model.fit(xs_train, ys_train, epochs=epochs, batch_size=128, callbacks=[cb],
              validation_data=(xs_test, ys_test), validation_batch_size=len(ys_test))
    model.load_weights('base.h5')
    model.evaluate(xs_test, ys_test, batch_size=len(ys_test))
    efn_base.save_weights('efn_base.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()

    os.chdir(args.dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    base(args.epochs, args.learning_rate)
