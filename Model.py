from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pathlib
import os
from Data import CLASS_NUM, INPUT_LENGTH, CHANNEL_NUM, BATCH_NUM

HIDDEN_UNIT_NUM = 1000
ALPHA = 0.2
SUMM1 = [True] * CHANNEL_NUM
SUMM2 = [True] * CHANNEL_NUM
SUMM3 = [True] * CHANNEL_NUM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def leaky_relu(s):
    return tf.nn.relu(s) - ALPHA * tf.nn.relu(-s)

def noise_generator(noise):
    with tf.variable_scope('Noise_generator'):
        noise = tf.expand_dims(input=noise, axis=1)
        # shape: [1, 1, NOISE_LENGTH]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, axis=1, name='batch_norm1', fused=True)
        conv2 = tf.layers.conv1d(inputs=batch_norm1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2, axis=1, name='batch_norm2', fused=True)
        conv3 = tf.layers.conv1d(inputs=batch_norm2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3, axis=1, name='batch_norm3', fused=True)
        conv4 = tf.layers.conv1d(inputs=batch_norm3, filters=BATCH_NUM, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=None, use_bias=False, name='conv4')
        # shape: [1, BATCH_NUM, NOISE_LENGTH]
        conv4 = tf.squeeze(input=conv4, axis=0)
        # shape: [BATCH_NUM, NOISE_LENGTH]
        return conv4

def time_seq_noise_generator(noise, num):
    with tf.variable_scope('Time_seq_noise_generator' + str(num)):
        noise = tf.expand_dims(input=noise, axis=1)
        # shape: [1, 1, NOISE_LENGTH]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, axis=1, name='batch_norm1', fused=True)
        conv2 = tf.layers.conv1d(inputs=batch_norm1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2, axis=1, name='batch_norm2', fused=True)
        conv3 = tf.layers.conv1d(inputs=batch_norm2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3, axis=1, name='batch_norm3', fused=True)
        conv4 = tf.layers.conv1d(inputs=batch_norm3, filters=BATCH_NUM, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=None, use_bias=False, name='conv4')
        # shape: [1, BATCH_NUM, NOISE_LENGTH]
        conv4 = tf.squeeze(input=conv4, axis=0)
        # shape: [BATCH_NUM, NOISE_LENGTH]
        return conv4

def generator1(noise, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        # shape: [BATCH_SIZE, NOISE_LENGTH]
        dense1 = tf.layers.dense(inputs=noise, units=CLASS_NUM * INPUT_LENGTH // 4, activation=leaky_relu, name='dense1')
        # shape: [BATCH_SIZE, CLASS_NUM * INPUT_LENGTH // 4]
        dropout1 = tf.layers.dropout(inputs=dense1, training=train, name='dropout1')
        reshaped_inputs = tf.reshape(dropout1, [-1, 64, CLASS_NUM // 8, INPUT_LENGTH // 32])
        # shape: [BATCH_SIZE, 64, CLASS_NUM // 8, INPUT_LENGTH // 32]
        deconv1 = tf.layers.conv2d_transpose(inputs=reshaped_inputs, filters=16, kernel_size=[5, 5], strides=(2, 2), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 16]
        batch_norm1 = tf.layers.batch_normalization(inputs=deconv1, axis=1, training=train, name='batch_norm1', fused=True)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm1, filters=4, kernel_size=[5, 5], strides=(1, 2), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv2')
        # shape: [BATCH_SIZE, 4, CLASS_NUM // 4, INPUT_LENGTH // 8]
        batch_norm2 = tf.layers.batch_normalization(inputs=deconv2, axis=1, training=train, name='batch_norm2', fused=True)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=1, kernel_size=[5, 5], strides=(1, 2), padding='same', data_format='channels_first', activation=None, use_bias=False, name='deconv3')
        # shape: [BATCH_SIZE, 1, CLASS_NUM // 4, INPUT_LENGTH // 4]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4]
        if SUMM1[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output[:BATCH_NUM // 10], -1))
            SUMM1[num] = False
        return output

def generator2(inputs, noise, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        dense1 = tf.layers.dense(inputs=noise, units=2 * CLASS_NUM // 8 * INPUT_LENGTH // 8, activation=leaky_relu, name='dense1')
        dense1 = tf.reshape(tensor=dense1, shape=[BATCH_NUM, 2, CLASS_NUM // 8, INPUT_LENGTH // 8])
        dense1 = tf.tile(input=dense1, multiples=[1, 1, 2, 2])
        inputs = tf.concat(values=[inputs, dense1], axis=1)
        # shape: [BATCH_SIZE, CHANNEL_NUM + 2, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        batch_norm1 = tf.layers.batch_normalization(inputs=conv2 + conv1, axis=1, training=train, name='batch_norm1', fused=True)
        deconv1 = tf.layers.conv2d_transpose(inputs=batch_norm1, filters=16, kernel_size=[5, 5], strides=(2, 2), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 2, INPUT_LENGTH // 2]
        batch_norm2 = tf.layers.batch_normalization(inputs=deconv1, axis=1, training=train, name='batch_norm2', fused=True)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=4, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv2')
        # shape: [BATCH_SIZE, 4, CLASS_NUM // 2, INPUT_LENGTH // 2]
        batch_norm3 = tf.layers.batch_normalization(inputs=deconv2, axis=1, training=train, name='batch_norm3', fused=True)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm3, filters=1, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=None, use_bias=False, name='deconv3')
        # shape: [BATCH_SIZE, 1, CLASS_NUM // 2, INPUT_LENGTH // 2]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2]
        if SUMM2[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output[:BATCH_NUM // 10], -1))
            SUMM2[num] = False
        return output

def generator3(inputs, noise, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        dense1 = tf.layers.dense(inputs=noise, units=2 * CLASS_NUM // 8 * INPUT_LENGTH // 8, activation=leaky_relu, name='dense1')
        dense1 = tf.reshape(tensor=dense1, shape=[BATCH_NUM, 2, CLASS_NUM // 8, INPUT_LENGTH // 8])
        dense1 = tf.tile(input=dense1, multiples=[1, 1, 4, 4])
        inputs = tf.concat(values=[inputs, dense1], axis=1)
        # shape: [BATCH_SIZE, CHANNEL_NUM + 2, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 2, INPUT_LENGTH // 2]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, axis=1, training=train, name='batch_norm1', fused=True)
        conv2 = tf.layers.conv2d(inputs=batch_norm1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2 + batch_norm1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        batch_norm2 = tf.layers.batch_normalization(inputs=conv3, axis=1, training=train, name='batch_norm2', fused=True)
        deconv1 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=16, kernel_size=[5, 5], strides=(2, 2), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM, INPUT_LENGTH]
        batch_norm3 = tf.layers.batch_normalization(inputs=deconv1, axis=1, training=train, name='batch_norm3', fused=True)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm3, filters=4, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='deconv2')
        # shape: [BATCH_SIZE, 4, CLASS_NUM, INPUT_LENGTH]
        batch_norm4 = tf.layers.batch_normalization(inputs=deconv2, axis=1, training=train, name='batch_norm4', fused=True)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm4, filters=1, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=None, use_bias=False, name='deconv3')
        # shape: [BATCH_SIZE, 1, CLASS_NUM, INPUT_LENGTH]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH]
        if SUMM3[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output[:BATCH_NUM // 10], -1))
            SUMM3[num] = False
        return output

def discriminator1(inputs, reuse=False):
    with tf.variable_scope('Discriminator1') as scope:
        if reuse:
            scope.reuse_variables()
        batch_norm1 = tf.layers.batch_normalization(inputs=inputs, axis=1, name='batch_norm1', fused=True)
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv1 = tf.layers.conv2d(inputs=batch_norm1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        pool1 = tf.layers.max_pooling2d(inputs=conv2 + conv1, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 8, INPUT_LENGTH // 8]
        batch_norm2 = tf.layers.batch_normalization(inputs=pool1, axis=1, name='batch_norm2', fused=True)
        conv3 = tf.layers.conv2d(inputs=batch_norm2, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 8, INPUT_LENGTH // 8]
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv4')
        pool2 = tf.layers.max_pooling2d(inputs=conv4 + conv3, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 16, INPUT_LENGTH // 16]
        batch_norm3 = tf.layers.batch_normalization(inputs=pool2, axis=1, name='batch_norm3', fused=True)
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=batch_norm3), units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator2(inputs, reuse=False):
    with tf.variable_scope('Discriminator2') as scope:
        if reuse:
            scope.reuse_variables()
        batch_norm1 = tf.layers.batch_normalization(inputs=inputs, axis=1, name='batch_norm1', fused=True)
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv1 = tf.layers.conv2d(inputs=batch_norm1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        pool1 = tf.layers.max_pooling2d(inputs=conv2 + conv1, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 4, INPUT_LENGTH // 4]
        batch_norm2 = tf.layers.batch_normalization(inputs=pool1, axis=1, name='batch_norm2', fused=True)
        conv3 = tf.layers.conv2d(inputs=batch_norm2, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv4')
        pool2 = tf.layers.max_pooling2d(inputs=conv4 + conv3, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 8, INPUT_LENGTH // 8]
        batch_norm3 = tf.layers.batch_normalization(inputs=pool2, axis=1, name='batch_norm3', fused=True)
        conv5 = tf.layers.conv2d(inputs=batch_norm3, filters=32, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv5')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 8, INPUT_LENGTH // 8]
        conv6 = tf.layers.conv2d(inputs=conv5, filters=32, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv6')
        pool3 = tf.layers.max_pooling2d(inputs=conv6 + conv5, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool3')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 16, INPUT_LENGTH // 8]
        batch_norm4 = tf.layers.batch_normalization(inputs=pool3, axis=1, name='batch_norm4', fused=True)
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=batch_norm4), units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator3(inputs, reuse=False):
    with tf.variable_scope('Discriminator3') as scope:
        if reuse:
            scope.reuse_variables()
        batch_norm1 = tf.layers.batch_normalization(inputs=inputs, axis=1, name='batch_norm1', fused=True)
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        conv1 = tf.layers.conv2d(inputs=batch_norm1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM, INPUT_LENGTH]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=8, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv2')
        pool1 = tf.layers.max_pooling2d(inputs=conv2 + conv1, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 2, INPUT_LENGTH // 2]
        batch_norm2 = tf.layers.batch_normalization(inputs=pool1, axis=1, name='batch_norm2', fused=True)
        conv3 = tf.layers.conv2d(inputs=batch_norm2, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv4')
        pool2 = tf.layers.max_pooling2d(inputs=conv4 + conv3, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        batch_norm3 = tf.layers.batch_normalization(inputs=pool2, axis=1, name='batch_norm3', fused=True)
        conv5 = tf.layers.conv2d(inputs=batch_norm3, filters=32, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv5')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv6 = tf.layers.conv2d(inputs=conv5, filters=32, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv6')
        pool3 = tf.layers.max_pooling2d(inputs=conv6 + conv5, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool3')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 8, INPUT_LENGTH // 8]
        batch_norm4 = tf.layers.batch_normalization(inputs=pool3, axis=1, name='batch_norm4', fused=True)
        conv7 = tf.layers.conv2d(inputs=batch_norm4, filters=64, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv7')
        # shape: [BATCH_SIZE, 64, CLASS_NUM // 8, INPUT_LENGTH // 8]
        conv8 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=[5, 5], padding='same', data_format='channels_first', activation=leaky_relu, use_bias=False, name='conv8')
        pool4 = tf.layers.max_pooling2d(inputs=conv8 + conv7, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='pool4')
        # shape: [BATCH_SIZE, 64, CLASS_NUM // 16, INPUT_LENGTH // 16]
        batch_norm5 = tf.layers.batch_normalization(inputs=pool4, axis=1, name='batch_norm5', fused=True)
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=batch_norm5), units=1, activation=tf.sigmoid, name='output')
    return output

def get_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)
