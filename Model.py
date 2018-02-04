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

def conv2d(inputs, filters, training, use_batch_norm=True, name=''):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[5, 5], strides=(1, 1), padding='same', data_format='channels_first', activation=leaky_relu, use_bias=not use_batch_norm, name='conv')
        if use_batch_norm:
            output = tf.layers.batch_normalization(inputs=conv, axis=1, training=training, name='batch_norm', fused=True)
        else:
            output = conv
        return output

def conv2d_transpose(inputs, filters, strides, training, use_batch_norm=True, name=''):
    with tf.variable_scope(name):
        deconv = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=[5, 5], strides=strides, padding='same', data_format='channels_first', activation=leaky_relu, use_bias=not use_batch_norm, name='deconv')
        if use_batch_norm:
            output = tf.layers.batch_normalization(inputs=deconv, axis=1, training=training, name='batch_norm', fused=True)
        else:
            output = deconv
        return output

def residual_block(inputs, filters, training, downsize=False, name=''):
    with tf.variable_scope(name):
        if inputs.get_shape()[1] != filters:
            inputs = conv2d(inputs=inputs, filters=filters, training=training, name='inputs')
        conv1 = conv2d(inputs=inputs, filters=filters, training=training, name='conv1')
        conv2 = conv2d(inputs=inputs, filters=filters, training=training, name='conv2')
        output = inputs + conv2
        if downsize:
            output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first')
        return output

def noise_generator(noise):
    with tf.variable_scope('Noise_generator'):
        noise = tf.expand_dims(input=noise, axis=1)
        # shape: [1, 1, NOISE_LENGTH]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        conv2 = tf.layers.conv1d(inputs=conv1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        conv3 = tf.layers.conv1d(inputs=conv2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv4')
        # shape: [1, 32, NOISE_LENGTH]
        conv5 = tf.layers.conv1d(inputs=conv4, filters=BATCH_NUM, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=None, name='conv5')
        # shape: [1, BATCH_NUM, NOISE_LENGTH]
        output = tf.squeeze(input=conv5, axis=0)
        # shape: [BATCH_NUM, NOISE_LENGTH]
        return output

def time_seq_noise_generator(noise, num):
    with tf.variable_scope('Time_seq_noise_generator' + str(num)):
        noise = tf.expand_dims(input=noise, axis=1)
        # shape: [1, 1, NOISE_LENGTH]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        conv2 = tf.layers.conv1d(inputs=conv1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        conv3 = tf.layers.conv1d(inputs=conv2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.selu, name='conv4')
        # shape: [1, 32, NOISE_LENGTH]
        conv5 = tf.layers.conv1d(inputs=conv4, filters=BATCH_NUM, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=None, name='conv5')
        # shape: [1, BATCH_NUM, NOISE_LENGTH]
        output = tf.squeeze(input=conv5, axis=0)
        # shape: [BATCH_NUM, NOISE_LENGTH]
        return output

def generator1(noise, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        # shape: [BATCH_SIZE, NOISE_LENGTH]
        dense1 = tf.layers.dense(inputs=noise, units=CLASS_NUM * INPUT_LENGTH // 4, activation=tf.nn.selu, name='dense1')
        # shape: [BATCH_SIZE, CLASS_NUM * INPUT_LENGTH // 4]
        dropout1 = tf.layers.dropout(inputs=dense1, training=train, name='dropout1')
        reshaped_inputs = tf.reshape(dropout1, [-1, 64, CLASS_NUM // 8, INPUT_LENGTH // 32])
        # shape: [BATCH_SIZE, 64, CLASS_NUM // 8, INPUT_LENGTH // 32]
        deconv1 = conv2d_transpose(inputs=reshaped_inputs, filters=16, strides=(2, 2), training=train, name='deconv1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv1 = conv2d(inputs=deconv1, filters=16, training=train, name='conv1')
        deconv2 = conv2d_transpose(inputs=conv1, filters=4, strides=(1, 2), training=train, name='deconv2')
        # shape: [BATCH_SIZE, 4, CLASS_NUM // 4, INPUT_LENGTH // 8]
        conv2 = conv2d(inputs=deconv2, filters=4, training=train, name='conv2')
        deconv3 = conv2d_transpose(inputs=conv2, filters=1, strides=(1, 2), training=train, use_batch_norm=False, name='deconv3')
        # shape: [BATCH_SIZE, 1, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv3 = conv2d(inputs=deconv3, filters=1, training=train, use_batch_norm=False, name='conv3')
        output = tf.squeeze(input=deconv3, axis=1)
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4]
        output_dis = tf.tanh(tf.squeeze(input=conv3, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4]
        if SUMM1[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output_dis[:BATCH_NUM // 10], -1))
            SUMM1[num] = False
        return output_dis, output

def generator2(inputs, noise, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        dense1 = tf.layers.dense(inputs=noise, units=2 * CLASS_NUM // 8 * INPUT_LENGTH // 8, activation=tf.nn.selu, name='dense1')
        dense1 = tf.reshape(tensor=dense1, shape=[BATCH_NUM, 2, CLASS_NUM // 8, INPUT_LENGTH // 8])
        dense1 = tf.tile(input=dense1, multiples=[1, 1, 2, 2])
        inputs = tf.concat(values=[inputs, dense1], axis=1)
        # shape: [BATCH_SIZE, CHANNEL_NUM + 2, CLASS_NUM // 4, INPUT_LENGTH //
        # 4]
        res1 = residual_block(inputs=inputs, filters=16, training=train, name='res1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        res2 = residual_block(inputs=res1, filters=4, training=train, name='res2')
        deconv1 = conv2d_transpose(inputs=res2, filters=1, strides=(2, 2), training=train, use_batch_norm=False, name='deconv1')
        # shape: [BATCH_SIZE, 4, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv2 = conv2d(inputs=deconv1, filters=1, training=train, use_batch_norm=False, name='conv2')
        # shape: [BATCH_SIZE, 1, CLASS_NUM // 2, INPUT_LENGTH // 2]
        output_dis = tf.tanh(tf.squeeze(input=conv2, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2]
        output = tf.squeeze(input=deconv1, axis=1)
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2]
        if SUMM2[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output_dis[:BATCH_NUM // 10], -1))
            SUMM2[num] = False
        return output_dis, output

def generator3(inputs, noise, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        dense1 = tf.layers.dense(inputs=noise, units=2 * CLASS_NUM // 4 * INPUT_LENGTH // 4, activation=tf.nn.selu, name='dense1')
        dense1 = tf.reshape(tensor=dense1, shape=[BATCH_NUM, 2, CLASS_NUM // 4, INPUT_LENGTH // 4])
        dense1 = tf.tile(input=dense1, multiples=[1, 1, 2, 2])
        inputs = tf.concat(values=[inputs, dense1], axis=1)
        # shape: [BATCH_SIZE, CHANNEL_NUM + 2, CLASS_NUM // 2, INPUT_LENGTH //
        # 2]
        res1 = residual_block(inputs=inputs, filters=16, training=train, name='res1')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 2, INPUT_LENGTH // 2]
        res2 = residual_block(inputs=res1, filters=4, training=train, name='res2')
        deconv1 = conv2d_transpose(inputs=res2, filters=1, strides=(2, 2), training=train, use_batch_norm=False, name='deconv1')
        # shape: [BATCH_SIZE, 4, CLASS_NUM, INPUT_LENGTH]
        conv2 = conv2d(inputs=deconv1, filters=1, training=train, use_batch_norm=False, name='conv2')
        # shape: [BATCH_SIZE, 1, CLASS_NUM, INPUT_LENGTH]
        output = tf.tanh(tf.squeeze(input=conv2, axis=1))
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH]
        if SUMM3[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output[:BATCH_NUM // 10], -1))
            SUMM3[num] = False
        return output

def discriminator1(inputs, train, reuse=False):
    with tf.variable_scope('Discriminator1') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        res1 = residual_block(inputs=inputs, filters=8, training=train, downsize=True, name='res1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 8, INPUT_LENGTH // 8]
        res2 = residual_block(inputs=res1, filters=16, training=train, downsize=True, name='res2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 16, INPUT_LENGTH // 16]
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=res2), units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator2(inputs, train, reuse=False):
    with tf.variable_scope('Discriminator2') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        res1 = residual_block(inputs=inputs, filters=8, training=train, downsize=True, name='res1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 4, INPUT_LENGTH // 4]
        res2 = residual_block(inputs=res1, filters=16, training=train, downsize=True, name='res2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 8, INPUT_LENGTH // 8]
        res3 = residual_block(inputs=res2, filters=32, training=train, downsize=True, name='res3')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 16, INPUT_LENGTH // 16]
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=res3), units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator3(inputs, train, reuse=False):
    with tf.variable_scope('Discriminator3') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        res1 = residual_block(inputs=inputs, filters=8, training=train, downsize=True, name='res1')
        # shape: [BATCH_SIZE, 8, CLASS_NUM // 2, INPUT_LENGTH // 2]
        res2 = residual_block(inputs=res1, filters=16, training=train, downsize=True, name='res2')
        # shape: [BATCH_SIZE, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        res3 = residual_block(inputs=res2, filters=32, training=train, downsize=True, name='res3')
        # shape: [BATCH_SIZE, 32, CLASS_NUM // 8, INPUT_LENGTH // 8]
        res4 = residual_block(inputs=res3, filters=64, training=train, downsize=True, name='res4')
        # shape: [BATCH_SIZE, 64, CLASS_NUM // 16, INPUT_LENGTH // 16]
        output = tf.layers.dense(inputs=tf.layers.flatten(inputs=res3), units=1, activation=tf.sigmoid, name='output')
    return output

def get_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)
