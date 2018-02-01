from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pathlib
import os
from Data import CLASS_NUM, INPUT_LENGTH, CHANNEL_NUM

HIDDEN_UNIT_NUM = 1000
ALPHA = 0.2
HIST = [True] * CHANNEL_NUM
SUMM = [[True] * CHANNEL_NUM] * 3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def leaky_relu(s):
    return tf.nn.relu(s) - ALPHA * tf.nn.relu(-s)

#def generator(noise, shared_noise, num, train):
#    with tf.variable_scope('generator' + str(num)):
#        inputs = tf.concat([noise, shared_noise], 1)
#        dense1 = tf.layers.dense(inputs=inputs, units=CLASS_NUM * INPUT_LENGTH
#        // 4, activation=leaky_relu, name='dense2')
#        dropout1 = tf.layers.dropout(inputs=dense1, training=train,
#        name='dropout2')
#        dropout1 = tf.reshape(dropout1, [-1, CLASS_NUM // 8, INPUT_LENGTH //
#        32, 64])
#        deconv1 = tf.layers.conv2d_transpose(inputs=dropout1, filters=32,
#        kernel_size=[12, 16], strides=(2, 2), padding='same',
#        activation=leaky_relu, name='deconv1', use_bias=False)
#        batch_norm1 = tf.layers.batch_normalization(inputs=deconv1)
#        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm1, filters=16,
#        kernel_size=[12, 16], strides=(2, 2), padding='same',
#        activation=leaky_relu, name='deconv2', use_bias=False)
#        batch_norm2 = tf.layers.batch_normalization(inputs=deconv2)
#        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=8,
#        kernel_size=[12, 16], strides=(2, 2), padding='same',
#        activation=leaky_relu, name='deconv3', use_bias=False)
#        batch_norm3 = tf.layers.batch_normalization(inputs=deconv3)
#        deconv4 = tf.layers.conv2d_transpose(inputs=batch_norm3, filters=4,
#        kernel_size=[12, 16], strides=(1, 2), padding='same',
#        activation=leaky_relu, name='deconv4', use_bias=False)
#        batch_norm4 = tf.layers.batch_normalization(deconv4)
#        deconv5 = tf.layers.conv2d_transpose(inputs=batch_norm4, filters=2,
#        kernel_size=[12, 16], strides=(1, 2), padding='same',
#        activation=leaky_relu, name='deconv5', use_bias=False)
#        batch_norm5 = tf.layers.batch_normalization(deconv5)
#        deconv6 = tf.layers.conv2d_transpose(inputs=batch_norm5, filters=1,
#        kernel_size=[12, 16], strides=(1, 1), padding='same', activation=None,
#        name='deconv6')
#        output = tf.tanh(tf.squeeze(deconv6, axis=3))
#        if num == 0 or num == 1:
#            with tf.variable_scope('deconv1', reuse=True):
#                deconv1_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv1_weight', deconv1_w)
#            with tf.variable_scope('deconv2', reuse=True):
#                deconv2_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv2_weight', deconv2_w)
#            with tf.variable_scope('deconv3', reuse=True):
#                deconv3_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv3_weight', deconv3_w)
#            with tf.variable_scope('deconv4', reuse=True):
#                deconv4_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv4_weight', deconv4_w)
#            with tf.variable_scope('deconv5', reuse=True):
#                deconv5_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv5_weight', deconv5_w)
#            with tf.variable_scope('deconv6', reuse=True):
#                deconv6_w = tf.get_variable('kernel')
#            tf.summary.histogram('deconv6_weight', deconv6_w)
#        if HIST[num]:
#            tf.summary.image('piano_roll', tf.expand_dims(output, -1))
#            HIST[num] = False
#    return output
def noise_generator(noise, length):
    with tf.variable_scope('noise_generator'):
        noise = tf.expand_dims(input=noise, axis=-1)
        # shape: [1, NOISE_LENGTH, 1]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [1, NOISE_LENGTH, 2]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, name='batch_norm1')
        conv2 = tf.layers.conv1d(inputs=batch_norm1, filters=4, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [1, NOISE_LENGTH, 4]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2, name='batch_norm2')
        conv3 = tf.layers.conv1d(inputs=batch_norm2, filters=8, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [1, NOISE_LENGTH, 8]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3, name='batch_norm3')
        conv4 = tf.layers.conv1d(inputs=batch_norm3, filters=length, kernel_size=5, strides=1, padding='same', activation=None, use_bias=False, name='conv4')
        # shape: [1, NOISE_LENGTH, length]
        conv4 = tf.transpose(x=conv4, perm=[2, 1, 0])
        return conv4

def time_seq_noise_generator(noise, num):
    with tf.variable_scope('time_seq_noise_generator' + str(num)):
        noise = tf.expand_dims(input=noise, axis=-1)
        # shape: [1, NOISE_LENGTH, 1]
        conv1 = tf.layers.conv1d(inputs=noise, filters=2, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [1, NOISE_LENGTH, 2]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, name='batch_norm1')
        conv2 = tf.layers.conv1d(inputs=batch_norm1, filters=4, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [1, NOISE_LENGTH, 4]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2, name='batch_norm2')
        conv3 = tf.layers.conv1d(inputs=batch_norm2, filters=8, kernel_size=5, strides=1, padding='same', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [1, NOISE_LENGTH, 8]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3, name='batch_norm3')
        conv4 = tf.layers.conv1d(inputs=batch_norm3, filters=length, kernel_size=5, strides=1, padding='same', activation=None, use_bias=False, name='conv4')
        # shape: [1, NOISE_LENGTH, length]
        conv4 = tf.transpose(x=conv4, perm=[2, 1, 0])
        return conv4

def generator1(noise, num, train):
    with tf.variable_scope('generator1_' + str(num)):
        # shape: [BATCH_SIZE, NOISE_LENGTH]
        dense1 = tf.layers.dense(inputs=noise, units=CLASS_NUM * INPUT_LENGTH // 4, activation=leaky_relu, name='dense1')
        # shape: [BATCH_SIZE, CLASS_NUM * INPUT_LENGTH // 4]
        dropout1 = tf.layers.dropout(inputs=dense1, training=train)
        reshaped_inputs = tf.reshape(dropout1, [-1, CLASS_NUM // 8, INPUT_LENGTH // 32, 64])
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 32, 64]
        deconv1 = tf.layers.conv2d_transpose(inputs=reshaped_inputs, filters=16, kernel_size=[12, 16], strides=(2, 2), padding='same', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 16, 16]
        batch_norm1 = tf.layers.batch_normalization(deconv1, training=train)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm1, filters=4, kernel_size=[12, 16], strides=(1, 2), padding='same', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 4]
        batch_norm2 = tf.layers.batch_normalization(deconv2, training=train)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm2, filters=1, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=None, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 1]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=3))
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8]
        if SUMM[0][num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output, -1))
            SUMM[0][num] = False
        return output

def generator2(inputs, num, train):
    with tf.variable_scope('generator2_' + str(num)):
        inputs = tf.expand_dims(inputs, [-1])
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 1]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 16]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, training=train)
        conv2 = tf.layers.conv2d(inputs=batch_norm1, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 16]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2 + batch_norm1, training=train)
        conv3 = tf.layers.conv2d(inputs=batch_norm2, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 16]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3 + batch_norm2, training=train)
        conv4 = tf.layers.conv2d(inputs=batch_norm3, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv4')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 16]
        batch_norm4 = tf.layers.batch_normalization(inputs=conv4 + batch_norm3, training=train)
        deconv1 = tf.layers.conv2d_transpose(inputs=batch_norm4, filters=16, kernel_size=[12, 16], strides=(2, 2), padding='same', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 4, 16]
        batch_norm5 = tf.layers.batch_normalization(deconv1, training=train)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm5, filters=4, kernel_size=[12, 16], strides=(1, 2), padding='same', activation=leaky_relu, use_bias=False, name='deconv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 4]
        batch_norm6 = tf.layers.batch_normalization(deconv2, training=train)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm6, filters=1, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=None, use_bias=False, name='deconv3')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=3))
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2]
        if SUMM[1][num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output, -1))
            SUMM[1][num] = False
        return output

def generator3(inputs, num, train):
    with tf.variable_scope('generator3_' + str(num)):
        inputs = tf.expand_dims(inputs, [-1])
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 16]
        batch_norm1 = tf.layers.batch_normalization(inputs=conv1, training=train)
        conv2 = tf.layers.conv2d(inputs=batch_norm1, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 16]
        batch_norm2 = tf.layers.batch_normalization(inputs=conv2 + batch_norm1, training=train)
        conv3 = tf.layers.conv2d(inputs=batch_norm2, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv3')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 16]
        batch_norm3 = tf.layers.batch_normalization(inputs=conv3 + batch_norm2, training=train)
        conv4 = tf.layers.conv2d(inputs=batch_norm1, filters=16, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='conv4')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 16]
        batch_norm4 = tf.layers.batch_normalization(inputs=conv4 + batch_norm3, training=train)
        deconv1 = tf.layers.conv2d_transpose(inputs=inputs, filters=16, kernel_size=[12, 16], strides=(2, 2), padding='same', activation=leaky_relu, use_bias=False, name='deconv1')
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH, 16]
        batch_norm5 = tf.layers.batch_normalization(deconv1, training=train)
        deconv2 = tf.layers.conv2d_transpose(inputs=batch_norm5, filters=4, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=leaky_relu, use_bias=False, name='deconv2')
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH, 4]
        batch_norm6 = tf.layers.batch_normalization(deconv2, training=train)
        deconv3 = tf.layers.conv2d_transpose(inputs=batch_norm6, filters=1, kernel_size=[12, 16], strides=(1, 1), padding='same', activation=None, use_bias=False, name='deconv3')
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH, 1]
        output = tf.tanh(tf.squeeze(input=deconv3, axis=3))
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH]
        if SUMM[2][num]:
            tf.summary.image(name='piano_roll', tensor=tf.expand_dims(output, -1))
            SUMM[2][num] = False
        return output

def discriminator1(inputs, reuse=False):
    with tf.variable_scope('discriminator1') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, CHANNEL_NUM]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 8, 8]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 16, 8]
        conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 16, 8]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 16, INPUT_LENGTH // 32, 16]
        output = tf.layers.dense(inputs=pool2, units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator2(inputs, reuse=False):
    with tf.variable_scope('discriminator2') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, CHANNEL_NUM]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv1')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 8]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4, 8]
        conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4, 16]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 8, 16]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv3')
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 8, 32]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 16, INPUT_LENGTH // 8, 32]
        output = tf.layers.dense(inputs=pool2, units=1, activation=tf.sigmoid, name='output')
    return output

def discriminator3(inputs, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH, CHANNEL_NUM]
        conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv1')
        # shape: [BATCH_SIZE, CLASS_NUM, INPUT_LENGTH, 8]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 8]
        conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv2')
        # shape: [BATCH_SIZE, CLASS_NUM // 2, INPUT_LENGTH // 2, 16]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4, 16]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv3')
        # shape: [BATCH_SIZE, CLASS_NUM // 4, INPUT_LENGTH // 4, 32]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 8, 32]
        conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[12, 16], padding='same', activation=leaky_relu, name='conv4')
        # shape: [BATCH_SIZE, CLASS_NUM // 8, INPUT_LENGTH // 8, 64]
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        # shape: [BATCH_SIZE, CLASS_NUM // 16, INPUT_LENGTH // 16, 64]
        output = tf.layers.dense(inputs=pool4, units=1, activation=tf.sigmoid, name='output')
    return output

#def discriminator(inputs, reuse=False):
#    with tf.variable_scope('discriminator') as scope:
#        if reuse:
#            scope.reuse_variables()
#        tf.summary.histogram('disc_input', inputs)
#        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[12,
#        8], padding='same', activation=leaky_relu, name='conv1')
#        pool1 = tf.layers.average_pooling2d(conv1, pool_size=[2, 2],
#        strides=2)
#        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[12,
#        8], padding='same', activation=leaky_relu, name='conv2')
#        pool2 = tf.layers.average_pooling2d(conv2, pool_size=[2, 2],
#        strides=2)
#        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[12,
#        8], padding='same', activation=leaky_relu, name='conv3')
#        pool3 = tf.layers.average_pooling2d(conv3, pool_size=[2, 2],
#        strides=2)
#        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[12,
#        8], padding='same', activation=leaky_relu, name='conv4')
#        pool4 = tf.layers.average_pooling2d(conv4, pool_size=[2, 2],
#        strides=2)
#        output = tf.layers.dense(inputs=pool4, units=1, activation=tf.sigmoid,
#        name='output')
#        output_scalar = tf.reduce_mean(output)
#        tf.summary.scalar('output_mean', output_scalar)
#    return output
def get_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)
