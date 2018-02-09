from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pathlib
import os
from Data import CLASS_NUM, INPUT_LENGTH, CHANNEL_NUM, BATCH_NUM

NOISE_LENGTH = 32
HIDDEN_UNIT_NUM = 1000
ALPHA = 0.2
SUMM1 = [True] * CHANNEL_NUM
SUMM2 = [True] * CHANNEL_NUM
SUMM3 = [True] * CHANNEL_NUM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv2d(inputs, filters, training, kernel_size=[3, 3], strides=(1, 1), use_batch_norm=True, name=''):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, use_bias=not use_batch_norm, name='conv')
        if use_batch_norm:
            output = tf.layers.batch_normalization(inputs=conv, axis=1, training=training, name='batch_norm', fused=True)
        else:
            output = conv
        return output

def conv2d_transpose(inputs, filters, strides, training, use_batch_norm=True, name=''):
    with tf.variable_scope(name):
        deconv = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=list(strides), strides=strides, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, use_bias=not use_batch_norm, name='deconv')
        if use_batch_norm:
            output = tf.layers.batch_normalization(inputs=deconv, axis=1, training=training, name='batch_norm', fused=True)
        else:
            output = deconv
        return output

def residual_block(inputs, filters, training, use_batch_norm=True, name=''):
    with tf.variable_scope(name):
        if inputs.get_shape()[1] != filters:
            inputs = conv2d(inputs=inputs, filters=filters, training=training, use_batch_norm=use_batch_norm, name='inputs')
        conv1 = conv2d(inputs=inputs, filters=filters, training=training, use_batch_norm=use_batch_norm, name='conv1')
        conv2 = conv2d(inputs=conv1, filters=filters, training=training, use_batch_norm=use_batch_norm, name='conv2')
        output = inputs + conv2
        return output

def noise_generator(noise):
    with tf.variable_scope('Noise_generator'):
        noise = tf.expand_dims(input=noise, axis=1)
        # shape: [1, 1, NOISE_LENGTH]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        conv2 = tf.layers.conv1d(inputs=conv1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        conv3 = tf.layers.conv1d(inputs=conv2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv4')
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
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv1')
        # shape: [1, 2, NOISE_LENGTH]
        conv2 = tf.layers.conv1d(inputs=conv1, filters=4, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv2')
        # shape: [1, 4, NOISE_LENGTH]
        conv3 = tf.layers.conv1d(inputs=conv2, filters=16, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv3')
        # shape: [1, 16, NOISE_LENGTH]
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=tf.nn.leaky_relu, name='conv4')
        # shape: [1, 32, NOISE_LENGTH]
        conv5 = tf.layers.conv1d(inputs=conv4, filters=BATCH_NUM, kernel_size=5, strides=1, padding='same', data_format='channels_first', activation=None, name='conv5')
        # shape: [1, BATCH_NUM, NOISE_LENGTH]
        output = tf.squeeze(input=conv5, axis=0)
        # shape: [BATCH_NUM, NOISE_LENGTH]
        return output

def encoder(inputs, train):
    with tf.variable_scope('Encoder'):
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        conv1 = conv2d(inputs=inputs, filters=16, training=train, kernel_size=[2, 1], strides=(2, 1), name='conv1')
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH]
        conv2 = conv2d(inputs=conv1, filters=16, training=train, kernel_size=[3, 1], strides=(3, 1), name='conv2')
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 6, INPUT_LENGTH]
        conv3 = conv2d(inputs=conv2, filters=16, training=train, kernel_size=[3, 1], strides=(3, 1), name='conv3')
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH]
        conv4 = conv2d(inputs=conv3, filters=16, training=train, kernel_size=[1, 3], strides=(1, 3), name='conv4')
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH // 3]
        conv5 = conv2d(inputs=conv4, filters=CHANNEL_NUM, training=train, kernel_size=[1, 4], strides=(1, 4), name='conv5')
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH // 12]
        output = tf.transpose(tf.reshape(conv5, [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]), [1, 0, 2])
        # shape: [CHANNEL_NUM, BATCH_NUM, NOISE_LENGTH * 4]
        return output

def generator1(noise, encode, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        noise = tf.concat([noise, encode], axis=1)
        noise = tf.reshape(tensor=noise, shape=[-1, NOISE_LENGTH * 8, 1, 1])
        # shape: [BATCH_NUM, NOISE_LENGTH * 8, 1, 1]
        deconv1 = conv2d_transpose(inputs=noise, filters=NOISE_LENGTH * 64, strides=(1, 2), training=train, name='deconv1')
        # shape: [BATCH_NUM, NOISE_LENGTH * 8, 1, 2]
        deconv2 = conv2d_transpose(inputs=deconv1, filters=NOISE_LENGTH * 32, strides=(1, 3), training=train, name='deconv2')
        # shape: [BATCH_NUM, NOISE_LENGTH * 4, 1, 6]
        deconv3 = conv2d_transpose(inputs=deconv2, filters=NOISE_LENGTH * 16, strides=(1, 4), training=train, name='deconv3')
        # shape: [BATCH_NUM, NOISE_LENGTH * 2, 1, 24]
        deconv4 = conv2d_transpose(inputs=deconv3, filters=NOISE_LENGTH * 8, strides=(1, 4), training=train, name='deconv4')
        # shape: [BATCH_NUM, NOISE_LENGTH, 1, 96]
        deconv5 = conv2d_transpose(inputs=deconv4, filters=NOISE_LENGTH * 4, strides=(2, 1), training=train, name='deconv5')
        # shape: [BATCH_NUM, NOISE_LENGTH // 2, 2, 96]
        deconv6 = conv2d_transpose(inputs=deconv5, filters=NOISE_LENGTH * 2, strides=(3, 1), training=train, name='deconv6')
        # shape: [BATCH_NUM, NOISE_LENGTH // 4, 6, 96]
        deconv7 = conv2d_transpose(inputs=deconv6, filters=NOISE_LENGTH, strides=(3, 1), training=train, name='deconv7')
        # shape: [BATCH_NUM, NOISE_LENGTH // 8, CLASS_NUM // 4, INPUT_LENGTH // 4]
        conv1 = conv2d(inputs=deconv7, filters=1, training=train, use_batch_norm=False, name='conv1')
        # shape: [BATCH_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        output_dis = tf.tanh(tf.squeeze(input=conv1, axis=1))
        # shape: [BATCH_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        if SUMM1[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output_dis[:BATCH_NUM // 10], -1))
            SUMM1[num] = False
        return output_dis, deconv7

def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        encode = tf.transpose(encode, [1, 0, 2])
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]
        encode = tf.layers.max_pooling1d(inputs=encode, pool_size=2, strides=2, padding='same', data_format='channels_first')
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 2]
        encode = tf.reshape(encode, [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 36, INPUT_LENGTH // 12])
        # shape: [BATCH_NUM, CHANNEL_NUM , CLASS_NUM // 36, INPUT_LENGTH // 12]
        encode = tf.tile(input=encode, multiples=(1, 1, 9, 3))
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        inputs = tf.concat([inputs, encode], axis=1)
        res1 = residual_block(inputs=inputs, filters=64, training=train, name='res1')
        # shape: [BATCH_NUM, 64, CLASS_NUM // 4, INPUT_LENGTH // 4]
        res2 = residual_block(inputs=res1, filters=32, training=train, name='res2')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 4, INPUT_LENGTH // 4]
        deconv1 = conv2d_transpose(inputs=res2, filters=16, strides=(2, 1), training=train, name='deconv1')
        # shape: [BATCH_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 4]
        deconv2 = conv2d_transpose(inputs=deconv1, filters=4, strides=(1, 2), training=train, name='deconv2')
        # shape: [BATCH_NUM, 1, CLASS_NUM // 2, INPUT_LENGTH // 2]
        conv1 = conv2d(inputs=deconv2, filters=1, training=train, use_batch_norm=False, name='conv1')
        output_dis = tf.tanh(tf.squeeze(input=conv1, axis=1))
        # shape: [BATCH_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        if SUMM2[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output_dis[:BATCH_NUM // 10], -1))
            SUMM2[num] = False
        return output_dis, deconv2

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        encode = tf.transpose(encode, [1, 0, 2])
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]
        encode = tf.reshape(encode, [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH // 12])
        # shape: [BATCH_NUM, CHANNEL_NUM , CLASS_NUM // 18, INPUT_LENGTH // 12]
        encode = tf.tile(input=encode, multiples=(1, 1, 9, 6))
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        inputs = tf.concat([inputs, encode], axis=1)
        res1 = residual_block(inputs=inputs, filters=64, training=train, name='res1')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 2, INPUT_LENGTH // 2]
        res2 = residual_block(inputs=res1, filters=16, training=train, name='res2')
        deconv1 = conv2d_transpose(inputs=res2, filters=4, strides=(2, 1), training=train, name='deconv1')
        # shape: [BATCH_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 4]
        deconv2 = conv2d_transpose(inputs=deconv1, filters=1, strides=(1, 2), training=train, use_batch_norm=False, name='deconv2')
        # shape: [BATCH_NUM, 1, CLASS_NUM // 2, INPUT_LENGTH // 2]
        output = tf.tanh(tf.squeeze(input=deconv2, axis=1))
        # shape: [BATCH_NUM, CLASS_NUM, INPUT_LENGTH]
        if SUMM3[num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output[:BATCH_NUM // 10], -1))
            SUMM3[num] = False
        return output

def discriminator1(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv2d(inputs=inputs, filters=8, kernel_size=[2, 1], training=train, strides=(2, 1), use_batch_norm=False, name='conv1')
        # shape: [BATCH_NUM, 8, CLASS_NUM // 8, INPUT_LENGTH // 4]
        conv2 = conv2d(inputs=conv1, filters=16, kernel_size=[2, 1], training=train, strides=(2, 1), use_batch_norm=False, name='conv2')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 16, INPUT_LENGTH // 4]
        conv3 = conv2d(inputs=conv2, filters=32, kernel_size=[1, 2], training=train, strides=(1, 2), use_batch_norm=False, name='conv3')
        # shape: [BATCH_NUM, 32, CLASS_NUM // 16, INPUT_LENGTH // 8]
        conv4 = conv2d(inputs=conv3, filters=64, kernel_size=[1, 2], training=train, strides=(1, 2), use_batch_norm=False, name='conv4')
        # shape: [BATCH_NUM, 64, CLASS_NUM // 16, INPUT_LENGTH // 16]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv4), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator1_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator1_Conditional', reuse=tf.AUTO_REUSE):
        encode = tf.transpose(encode, [1, 0, 2])
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]
        encode = tf.layers.max_pooling1d(inputs=encode, pool_size=2, strides=2, padding='same', data_format='channels_first')
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 2]
        encode = tf.reshape(encode, [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 36, INPUT_LENGTH // 12])
        # shape: [BATCH_NUM, CHANNEL_NUM , CLASS_NUM // 36, INPUT_LENGTH // 12]
        encode = tf.tile(input=encode, multiples=(1, 1, 9, 3))
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        output = (discriminator1(inputs=inputs, train=train, name='Discriminator1_Uncond') + discriminator1(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator1_Cond')) / 2.0
    return output

def discriminator2(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv2d(inputs=inputs, filters=8, kernel_size=[2, 1], training=train, strides=(2, 1), use_batch_norm=False, name='conv1')
        # shape: [BATCH_NUM, 8, CLASS_NUM // 4, INPUT_LENGTH // 2]
        conv2 = conv2d(inputs=conv1, filters=16, kernel_size=[4, 1], training=train, strides=(4, 1), use_batch_norm=False, name='conv2')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 16, INPUT_LENGTH // 2]
        conv3 = conv2d(inputs=conv2, filters=32, kernel_size=[1, 2], training=train, strides=(1, 2), use_batch_norm=False, name='conv3')
        # shape: [BATCH_NUM, 32, CLASS_NUM // 16, INPUT_LENGTH // 4]
        conv4 = conv2d(inputs=conv3, filters=64, kernel_size=[1, 4], training=train, strides=(1, 4), use_batch_norm=False, name='conv4')
        # shape: [BATCH_NUM, 64, CLASS_NUM // 16, INPUT_LENGTH // 16]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv4), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator2_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator2_Conditional', reuse=tf.AUTO_REUSE):
        encode = tf.transpose(encode, [1, 0, 2])
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]
        encode = tf.reshape(encode, [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH // 12])
        # shape: [BATCH_NUM, CHANNEL_NUM , CLASS_NUM // 18, INPUT_LENGTH // 12]
        encode = tf.tile(input=encode, multiples=(1, 1, 9, 6))
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        output = (discriminator2(inputs=inputs, train=train, name='Discriminator2_Uncond') + discriminator2(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator2_Cond')) / 2.0
    return output

def discriminator3(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv2d(inputs=inputs, filters=8, kernel_size=[4, 1], training=train, strides=(4, 1), use_batch_norm=False, name='conv1')
        # shape: [BATCH_NUM, 8, CLASS_NUM // 4, INPUT_LENGTH]
        conv2 = conv2d(inputs=conv1, filters=16, kernel_size=[4, 1], training=train, strides=(4, 1), use_batch_norm=False, name='conv2')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 16, INPUT_LENGTH]
        conv3 = conv2d(inputs=conv2, filters=32, kernel_size=[1, 4], training=train, strides=(1, 4), use_batch_norm=False, name='conv3')
        # shape: [BATCH_NUM, 32, CLASS_NUM // 16, INPUT_LENGTH // 4]
        conv4 = conv2d(inputs=conv3, filters=64, kernel_size=[1, 4], training=train, strides=(1, 4), use_batch_norm=False, name='conv4')
        # shape: [BATCH_NUM, 64, CLASS_NUM // 16, INPUT_LENGTH // 16]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv4), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
    return output


def discriminator3_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator3_Conditional', reuse=tf.AUTO_REUSE):
        encode = tf.transpose(encode, [1, 0, 2])
        # shape: [BATCH_NUM, CHANNEL_NUM, NOISE_LENGTH * 4]
        encode = tf.reshape(encode, [BATCH_NUM, CHANNEL_NUM, CLASS_NUM // 18, INPUT_LENGTH // 12])
        # shape: [BATCH_NUM, CHANNEL_NUM , CLASS_NUM // 18, INPUT_LENGTH // 12]
        encode = tf.tile(input=encode, multiples=(1, 1, 18, 12))
        # shape: [BATCH_NUM, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        output = (discriminator3(inputs=inputs, train=train, name='Discriminator3_Uncond') + discriminator3(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator3_Cond')) / 2.0
    return output

def get_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)
