from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pathlib
import os

class_num = 84
input_length = 512
hidden_unit_num = 1000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def leaky_relu(s):
    alpha = tf.get_variable(name='alpha', initializer=tf.constant(0.3, dtype=tf.float32), dtype=tf.float32)
    return tf.nn.relu(s) - alpha * tf.nn.relu(-s)

def generator(noise, sharednoise, num, train):
    with tf.variable_scope('generator' + str(num)):
        inputs = tf.concat([noise[num], sharednoise], 1)
        dense1 = tf.layers.dense(inputs=inputs, units=hidden_unit_num, activation=leaky_relu, name='dense1')
        dropout1 = tf.layers.dropout(dense1, training=train, name='dropout1')
        dense2 = tf.layers.dense(inputs=dropout1, units=class_num * input_length // 4, activation=leaky_relu, name='dense2')
        dropout2 = tf.layers.dropout(inputs=dense2, training=train, name='dropout2')
        dropout2 = tf.reshape(dropout2, [-1, class_num // 12, input_length // 32, 96])
        deconv1 = tf.layers.conv2d_transpose(inputs=dropout2, filters=64, kernel_size=[2, 4], strides=(2, 2), padding='same', activation=leaky_relu, name='deconv1', bias_initializer=None)
        batch_norm1 = tf.layers.batch_normalization(inputs=deconv1)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm1, filters=32, kernel_size=[4, 6], strides=(2, 2), padding='same', activation=leaky_relu, name='deconv2', bias_initializer=None)
        batch_norm2 = tf.layers.batch_normalization(inputs=deconv2)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=16, kernel_size=[6, 8], strides=(3, 2), padding='same', activation=leaky_relu, name='deconv3', bias_initializer=None)
        batch_norm3 = tf.layers.batch_normalization(inputs=deconv3)
        deconv4 = tf.layers.conv2d_transpose(inputs=batch_norm3, filters=8, kernel_size=[12, 8], strides=(1, 2), padding='same', activation=leaky_relu, name='deconv4', bias_initializer=None)
        batch_norm4 = tf.layers.batch_normalization(deconv4)
        deconv5 = tf.layers.conv2d_transpose(inputs=batch_norm4, filters=1, kernel_size=[12, 8], strides=(1, 2), padding='same', activation=leaky_relu, name='deconv5', bias_initializer=None)
        batch_norm5 = tf.layers.batch_normalization(deconv5)
        output = tf.tanh(tf.squeeze(batch_norm5, axis=3))
        if num == 0 or num == 1:
            with tf.variable_scope('deconv1', reuse=True):
                deconv1_w = tf.get_variable('kernel')
            tf.summary.histogram('deconv1_weight', deconv1_w)
            with tf.variable_scope('deconv2', reuse=True):
                deconv2_w = tf.get_variable('kernel')
            tf.summary.histogram('deconv2_weight', deconv2_w)
            with tf.variable_scope('deconv3', reuse=True):
                deconv3_w = tf.get_variable('kernel')
            tf.summary.histogram('deconv3_weight', deconv3_w)
            with tf.variable_scope('deconv4', reuse=True):
                deconv4_w = tf.get_variable('kernel')
            tf.summary.histogram('deconv4_weight', deconv4_w)
            with tf.variable_scope('deconv5', reuse=True):
                deconv5_w = tf.get_variable('kernel')
            tf.summary.histogram('deconv5_weight', deconv5_w)
    return output

def discriminator(inputs, reuse=False, train=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        tf.summary.histogram('disc_input', inputs)
        conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[12, 8], padding='same', activation=leaky_relu, name='conv1')
        pool1 = tf.layers.average_pooling2d(conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[12, 8], padding='same', activation=leaky_relu, name='conv2')
        pool2 = tf.layers.average_pooling2d(conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[12, 8], padding='same', activation=leaky_relu, name='conv3')
        pool3 = tf.layers.average_pooling2d(conv3, pool_size=[2, 2], strides=2)
        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[12, 8], padding='same', activation=leaky_relu, name='conv4')
        pool4 = tf.layers.average_pooling2d(conv4, pool_size=[2, 2], strides=2)
        dense1 = tf.layers.dense(inputs=tf.contrib.layers.flatten(pool4), units=hidden_unit_num // 10, activation=tf.nn.sigmoid, name='dense1')
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
        output_scalar = tf.reduce_mean(output)
        tf.summary.scalar('output_mean', output_scalar)
    return output

def get_noise(n_batch, n_noise):
    return np.random.normal(loc=0.0, scale=1.0, size=[n_batch, n_noise])