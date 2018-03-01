from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from Data import CLASS_NUM, INPUT_LENGTH, BATCH_NUM

NOISE_LENGTH = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv2d(inputs, filters, kernel_size=[3, 3], strides=(1, 1), training=True, regularization='', name=''):
    with tf.variable_scope(name):
        if regularization == 'selu':
            activation_function = tf.nn.selu
        else:
            activation_function = tf.nn.leaky_relu
        use_bias = regularization != 'batch_norm'
        output = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first', activation=activation_function, use_bias=use_bias, name='conv')
        if not use_bias:
            output = tf.layers.batch_normalization(inputs=output, axis=1, training=training, name='batch_norm', fused=True)
        return output

def convolution(inputs, filters, kernel_size=[1, 3, 3], strides=(1, 1, 1), training=True, regularization='', transpose=False, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if regularization == 'selu':
            activation_function = tf.nn.selu
        else:
            activation_function = tf.nn.leaky_relu
        use_bias = regularization != 'batch_norm'
        if kernel_size[0] != 1:
            if transpose:
                conv_func = tf.layers.conv3d_transpose
            else:
                conv_func = tf.layers.conv3d
            output = conv_func(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first', activation=activation_function, use_bias=use_bias, name='conv')
        else:
            if transpose:
                conv_func = tf.layers.conv2d_transpose
            else:
                conv_func = tf.layers.conv2d
            axis = 2
            kernel_size = kernel_size[1:]
            strides = strides[1:]
            inputs = tf.unstack(inputs, axis=axis, name='unstack')
            output = [conv_func(inputs=i, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first', activation=activation_function, use_bias=use_bias, name='conv') for i in inputs]
            output = tf.stack(output, axis=axis, name='stack')
        if not use_bias:
            output = tf.layers.batch_normalization(inputs=output, axis=1, training=training, name='batch_norm', fused=True)
        return output

def residual_block(inputs, filters, training, regularization='', name=''):
    with tf.variable_scope(name):
        if inputs.get_shape()[1] != filters:
            inputs = convolution(inputs=inputs, filters=filters, training=training, regularization=regularization, name='inputs')
        conv1 = convolution(inputs=inputs, filters=filters, training=training, regularization=regularization, name='conv1')
        conv2 = convolution(inputs=conv1, filters=filters, training=training, regularization=regularization, name='conv2')
        output = inputs + conv2
        return output

def noise_generator(noise, train):
    with tf.variable_scope('Noise_generator'):
        # shape: [None, 1, 1, NOISE_LENGTH]
        conv1 = conv2d(inputs=noise, filters=64, kernel_size=[1, 3], strides=(1, 1), training=train, regularization='batch_norm', name='conv1')
        # shape: [None, 64, 1, NOISE_LENGTH]
        conv2 = conv2d(inputs=conv1, filters=4, kernel_size=[1, 3], strides=(1, 1), training=train, regularization='batch_norm', name='conv2')
        # shape: [None, 4, 1, NOISE_LENGTH]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, NOISE_LENGTH]
        return output

def time_seq_noise_generator(noise, num, train):
    with tf.variable_scope('Time_seq_noise_generator' + str(num)):
        # shape: [None, 1, 1, NOISE_LENGTH]
        conv1 = conv2d(inputs=noise, filters=64, kernel_size=[1, 3], strides=(1, 1), training=train, regularization='batch_norm', name='conv1')
        # shape: [None, 64, 1, NOISE_LENGTH]
        conv2 = conv2d(inputs=conv1, filters=4, kernel_size=[1, 3], strides=(1, 1), training=train, regularization='batch_norm', name='conv2')
        # shape: [None, 4, 1, NOISE_LENGTH]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, NOISE_LENGTH]
        return output

def encoder(inputs, num, train):
    with tf.variable_scope('Encoder' + str(num)):
        # shape: [None, 1, 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = convolution(inputs=inputs, filters=16, training=train, kernel_size=[1, 3, 1], strides=(1, 3, 1), regularization='batch_norm', name='conv1')
        # shape: [None, 16, 4, CLASS_NUM // 3, INPUT_LENGTH // 4]
        conv2 = convolution(inputs=conv1, filters=16, training=train, kernel_size=[1, 4, 1], strides=(1, 4, 1), regularization='batch_norm', name='conv2')
        # shape: [None, 16, 4, CLASS_NUM // 12, INPUT_LENGTH // 4]
        conv3 = convolution(inputs=conv2, filters=16, training=train, kernel_size=[1, 6, 1], strides=(1, 6, 1), regularization='batch_norm', name='conv3')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 4]
        conv4 = convolution(inputs=conv3, filters=16, training=train, kernel_size=[1, 1, 2], strides=(1, 1, 2), regularization='batch_norm', name='conv4')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 8]
        conv5 = convolution(inputs=conv4, filters=16, training=train, kernel_size=[1, 1, 3], strides=(1, 1, 3), regularization='batch_norm', name='conv5')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 24]
        conv6 = convolution(inputs=conv5, filters=16, training=train, kernel_size=[1, 1, 4], strides=(1, 1, 4), regularization='batch_norm', name='conv6')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 96]
        conv7 = convolution(inputs=conv6, filters=16, training=train, kernel_size=[1, 1, 4], strides=(1, 1, 4), regularization='batch_norm', name='conv7')
        # shape: [None, 16, 4, 1, 1]
        output = tf.transpose(tf.squeeze(conv7, axis=-1), perm=[0, 3, 2, 1])
        # shape: [None, 1, 4, 16]
        return output

def generator1(noise, encode, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        noise = tf.transpose(tf.concat([noise, encode], axis=2), perm=[0, 2, 1])
        # shape: [None, NOISE_LENGTH * 4 + 16, 4]
        noise = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, NOISE_LENGTH * 4 + 16, 4, 1, 1]
        deconv1 = convolution(inputs=noise, filters=NOISE_LENGTH * 64, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, regularization='batch_norm', transpose=True, name='deconv1')
        # shape: [None, NOISE_LENGTH * 64, 4, 1, 2]
        deconv2 = convolution(inputs=deconv1, filters=NOISE_LENGTH * 32, kernel_size=[1, 1, 3], strides=(1, 1, 3), training=train, regularization='batch_norm', transpose=True, name='deconv2')
        # shape: [None, NOISE_LENGTH * 32, 4, 1, 6]
        deconv3 = convolution(inputs=deconv2, filters=NOISE_LENGTH * 16, kernel_size=[1, 1, 4], strides=(1, 1, 4), training=train, regularization='batch_norm', transpose=True, name='deconv3')
        # shape: [None, NOISE_LENGTH * 16, 4, 1, 24]
        deconv4 = convolution(inputs=deconv3, filters=NOISE_LENGTH * 8, kernel_size=[1, 2, 1], strides=(1, 2, 1), training=train, regularization='batch_norm', transpose=True, name='deconv4')
        # shape: [None, NOISE_LENGTH * 8, 4, 2, 24]
        deconv5 = convolution(inputs=deconv4, filters=NOISE_LENGTH * 4, kernel_size=[1, 3, 1], strides=(1, 3, 1), training=train, regularization='batch_norm', transpose=True, name='deconv5')
        # shape: [None, NOISE_LENGTH * 4, 4, 6, 24]
        deconv6 = convolution(inputs=deconv5, filters=NOISE_LENGTH * 2, kernel_size=[1, 3, 1], strides=(1, 3, 1), training=train, regularization='batch_norm', transpose=True, name='deconv6')
        # shape: [None, NOISE_LENGTH * 2, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv1 = convolution(inputs=deconv6, filters=1, training=train, regularization='batch_norm', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        output = tf.tanh(tf.transpose(conv1, perm=[0, 2, 3, 4, 1]))
        # shape: [None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16, 1]
        tf.summary.image(name='piano_roll', tensor=tf.concat([output[:BATCH_NUM // 10, i] for i in range(4)], axis=2))
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        return output, deconv6

def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        encode = tf.expand_dims(tf.expand_dims(tf.transpose(encode, perm=[0, 2, 1]), axis=-1), axis=-1)
        # shape: [None, 16, 4, 1, 1]
        encode = tf.tile(input=encode, multiples=(1, 4, 1, CLASS_NUM // 4, INPUT_LENGTH // 16))
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        inputs = tf.concat([inputs, encode], axis=1)
        # shape: [None, 128, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        res1 = residual_block(inputs=inputs, filters=64, training=train, regularization='batch_norm', name='res1')
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        res2 = residual_block(inputs=res1, filters=64, training=train, regularization='batch_norm', name='res2')
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        deconv1 = convolution(inputs=res2, filters=32, kernel_size=[1, 2, 1], strides=(1, 2, 1), training=train, regularization='batch_norm', transpose=True, name='deconv1')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 16]
        deconv2 = convolution(inputs=deconv1, filters=32, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, regularization='batch_norm', transpose=True, name='deconv2')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv1 = convolution(inputs=deconv2, filters=1, training=train, regularization='batch_norm', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        output = tf.tanh(tf.transpose(conv1, perm=[0, 2, 3, 4, 1]))
        # shape: [None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8, 1]
        tf.summary.image(name='piano_roll', tensor=tf.concat([output[:BATCH_NUM // 10, i] for i in range(4)], axis=2))
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        return output, deconv2

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        encode = tf.expand_dims(tf.expand_dims(tf.transpose(encode, perm=[0, 2, 1]), axis=-1), axis=-1)
        # shape: [None, 16, 4, 1, 1]
        encode = tf.tile(input=encode, multiples=(1, 2, 1, CLASS_NUM // 2, INPUT_LENGTH // 8))
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        inputs = tf.concat([inputs, encode], axis=1)
        # shape: [None, 64, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        res1 = residual_block(inputs=inputs, filters=32, training=train, regularization='batch_norm', name='res1')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        res2 = residual_block(inputs=res1, filters=32, training=train, regularization='batch_norm', name='res2')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        deconv1 = convolution(inputs=res2, filters=16, kernel_size=[1, 2, 1], strides=(1, 2, 1), training=train, regularization='batch_norm', transpose=True, name='deconv1')
        # shape: [None, 16, 4, CLASS_NUM, INPUT_LENGTH // 8]
        deconv2 = convolution(inputs=deconv1, filters=4, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, regularization='batch_norm', transpose=True, name='deconv2')
        # shape: [None, 4, 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = convolution(inputs=deconv2, filters=1, training=train, regularization='batch_norm', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM, INPUT_LENGTH // 4]
        output = tf.tanh(tf.transpose(conv1, perm=[0, 2, 3, 4, 1]))
        # shape: [None, 4, CLASS_NUM, INPUT_LENGTH // 4, 1]
        tf.summary.image(name='piano_roll', tensor=tf.concat([output[:BATCH_NUM // 10, i] for i in range(4)], axis=2))
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM, INPUT_LENGTH // 4]
        return output

def discriminator1(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv1 = convolution(inputs=inputs, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), training=train, name='conv1')
        # shape: [None, 128, 2, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv2 = convolution(inputs=conv1, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), training=train, name='conv2')
        # shape: [None, 128, 1, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv3 = convolution(inputs=conv2, filters=128, kernel_size=[1, 3, 1], strides=(1, 3, 1), training=train, name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 12, INPUT_LENGTH // 16]
        conv4 = convolution(inputs=conv3, filters=128, kernel_size=[1, 1, 6], strides=(1, 6, 1), training=train, name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 16]
        conv5 = convolution(inputs=conv4, filters=128, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 32]
        conv6 = convolution(inputs=conv5, filters=256, kernel_size=[1, 1, 4], strides=(1, 1, 2), training=train, name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 64]
        conv7 = convolution(inputs=conv6, filters=512, kernel_size=[1, 1, 3], strides=(1, 1, 2), training=train, name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, name='output')
        # shape: [None, 1]
        return output

def discriminator1_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator1_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.stack([encode[:, :, :, 2 * i:2 * (i + 1)] for i in range(8)], axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 9, 3))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        uncond = discriminator1(inputs=inputs, train=train, name='Discriminator1_Uncond')
        cond = discriminator1(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator1_Cond')
        output = (uncond + cond) / 2.0
        return output

def discriminator2(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv1 = convolution(inputs=inputs, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), training=train, name='conv1')
        # shape: [None, 128, 2, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv2 = convolution(inputs=conv1, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), training=train, name='conv2')
        # shape: [None, 128, 1, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv3 = convolution(inputs=conv2, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), training=train, name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 12, INPUT_LENGTH // 8]
        conv4 = convolution(inputs=conv3, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), training=train, name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 8]
        conv5 = convolution(inputs=conv4, filters=128, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 16]
        conv6 = convolution(inputs=conv5, filters=256, kernel_size=[1, 1, 3], strides=(1, 1, 2), training=train, name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 32]
        conv7 = convolution(inputs=conv6, filters=512, kernel_size=[1, 1, 4], strides=(1, 1, 4), training=train, name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
        # shape: [None, 1]
        return output

def discriminator2_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator2_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.stack([encode[:, :, :, 2 * i:2 * (i + 1)] for i in range(8)], axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 18, 6))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        uncond = discriminator2(inputs=inputs, train=train, name='Discriminator2_Uncond')
        cond = discriminator2(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator2_Cond')
        output = (uncond + cond) / 2.0
        return output

def discriminator3(inputs, train, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = convolution(inputs=inputs, filters=128, kernel_size=[2, 1, 1], training=train, strides=(2, 1, 1), name='conv1')
        # shape: [None, 128, 2, CLASS_NUM, INPUT_LENGTH // 4]
        conv2 = convolution(inputs=conv1, filters=128, kernel_size=[2, 1, 1], training=train, strides=(2, 1, 1), name='conv2')
        # shape: [None, 128, 1, CLASS_NUM, INPUT_LENGTH // 4]
        conv3 = convolution(inputs=conv2, filters=128, kernel_size=[1, 6, 1], training=train, strides=(1, 6, 1), name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 6, INPUT_LENGTH // 4]
        conv4 = convolution(inputs=conv3, filters=128, kernel_size=[1, 12, 1], training=train, strides=(1, 12, 1), name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 4]
        conv5 = convolution(inputs=conv4, filters=128, kernel_size=[1, 1, 2], training=train, strides=(1, 1, 2), name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 8]
        conv6 = convolution(inputs=conv5, filters=256, kernel_size=[1, 1, 4], training=train, strides=(1, 1, 4), name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 32]
        conv7 = convolution(inputs=conv6, filters=512, kernel_size=[1, 1, 4], training=train, strides=(1, 1, 4), name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, activation=tf.sigmoid, name='output')
        # shape: [None, 1]
        return output


def discriminator3_conditional(inputs, encode, train):
    with tf.variable_scope('Discriminator3_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.stack([encode[:, :, :, 4 * i:4 * (i + 1)] for i in range(4)], axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 4, 4]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 18, 24))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        uncond = discriminator3(inputs=inputs, train=train, name='Discriminator3_Uncond')
        cond = discriminator3(inputs=tf.concat(values=[inputs, encode], axis=1), train=train, name='Discriminator3_Cond')
        output = (uncond + cond) / 2.0
        return output

def get_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)
