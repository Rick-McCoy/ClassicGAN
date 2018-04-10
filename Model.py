from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal
from Data import CLASS_NUM, INPUT_LENGTH, BATCH_SIZE

NOISE_LENGTH = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv(inputs, filters, kernel_size=[1, 3, 3], strides=(1, 1, 1), training=True, regularization='', transpose=False, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if regularization == 'batch_norm_relu':
            activation_function = tf.nn.relu
        elif regularization == 'batch_norm_tanh':
            activation_function = tf.tanh
        else:
            activation_function = tf.nn.leaky_relu
        use_bias = regularization == ''
        if inputs.get_shape().ndims == 4:
            if len(kernel_size) == 3:
                kernel_size = [3, 3]
                strides = (1, 1)
            if transpose:
                conv_func = tf.layers.conv2d_transpose
            else:
                conv_func = tf.layers.conv2d
            output = conv_func(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, \
                                padding='same', data_format='channels_first', activation=activation_function, \
                                use_bias=use_bias, name='conv')
        else:
            if transpose:
                conv_func = tf.layers.conv3d_transpose
            else:
                conv_func = tf.layers.conv3d
            output = conv_func(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, \
                                padding='same', data_format='channels_first', activation=activation_function, \
                                use_bias=use_bias, name='conv')
        if regularization != '':
            output = tf.layers.batch_normalization(inputs=output, axis=1, training=training, name='batch_norm', fused=True)
        return output

def residual_block(inputs, filters, training, regularization='', name=''):
    with tf.variable_scope(name):
        if inputs.get_shape().as_list()[1] != filters:
            inputs = conv(inputs=inputs, filters=filters, training=training, regularization=regularization, name='inputs')
        conv1 = conv(inputs=inputs, filters=filters, training=training, regularization=regularization, name='conv1')
        conv2 = conv(inputs=conv1, filters=filters, training=training, regularization=regularization, name='conv2')
        return inputs + conv2

def noise_generator(noise, train):
    with tf.variable_scope('Noise_generator'):
        # shape: [None, 1, 1, NOISE_LENGTH]
        conv1 = conv(inputs=noise, filters=1024, kernel_size=[1, 3], strides=(1, 1), training=train, \
                                                        regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1024, 1, NOISE_LENGTH]
        conv2 = conv(inputs=conv1, filters=4, kernel_size=[1, 3], strides=(1, 1), training=train, \
                                                        regularization='batch_norm_relu', name='conv2')
        # shape: [None, 4, 1, NOISE_LENGTH]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, NOISE_LENGTH]
        return output

def time_seq_noise_generator(noise, num, train):
    with tf.variable_scope('Time_seq_noise_generator' + str(num)):
        # shape: [None, 1, 1, NOISE_LENGTH]
        conv1 = conv(inputs=noise, filters=1024, kernel_size=[1, 3], strides=(1, 1), training=train, \
                                                        regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1024, 1, NOISE_LENGTH]
        conv2 = conv(inputs=conv1, filters=4, kernel_size=[1, 3], strides=(1, 1), training=train, \
                                                        regularization='batch_norm_relu', name='conv2')
        # shape: [None, 4, 1, NOISE_LENGTH]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, NOISE_LENGTH]
        return output

def encoder(inputs, num, train):
    with tf.variable_scope('Encoder' + str(num), reuse=tf.AUTO_REUSE):
        # shape: [None, 1, 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = conv(inputs=inputs, filters=16, kernel_size=[1, 3, 1], strides=(1, 3, 1), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv1')
        # shape: [None, 16, 4, CLASS_NUM // 3, INPUT_LENGTH // 4]
        conv2 = conv(inputs=conv1, filters=16, kernel_size=[1, 4, 1], strides=(1, 4, 1), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv2')
        # shape: [None, 16, 4, CLASS_NUM // 12, INPUT_LENGTH // 4]
        conv3 = conv(inputs=conv2, filters=16, kernel_size=[1, 6, 1], strides=(1, 6, 1), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv3')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 4]
        conv4 = conv(inputs=conv3, filters=16, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv4')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 8]
        conv5 = conv(inputs=conv4, filters=16, kernel_size=[1, 1, 3], strides=(1, 1, 3), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv5')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 24]
        conv6 = conv(inputs=conv5, filters=16, kernel_size=[1, 1, 4], strides=(1, 1, 4), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv6')
        # shape: [None, 16, 4, 1, INPUT_LENGTH // 96]
        conv7 = conv(inputs=conv6, filters=16, kernel_size=[1, 1, 4], strides=(1, 1, 4), training=train, \
                                                            regularization='batch_norm_lrelu', name='conv7')
        # shape: [None, 16, 4, 1, 1]
        flatten = [tf.layers.flatten(i) for i in tf.unstack(conv7, axis=2)]
        # shape: [4, None, 16]
        output_mean = [tf.layers.dense(i, units=16, activation=None, name='dense_mean') for i in flatten]
        # shape: [4, None, 16]
        output_mean = tf.stack(output_mean, axis=1)
        # shape: [None, 4, 16]
        output_mean = tf.expand_dims(output_mean, axis=1)
        # shape: [None, 1, 4, 16]
        output_var = [tf.layers.dense(i, units=16, activation=None, name='dense_var') for i in flatten]
        # shape: [4, None, 16]
        output_var = tf.stack(output_var, axis=1)
        # shape: [None, 4, 16]
        output_var = tf.expand_dims(output_var, axis=1)
        # shape: [None, 1, 4, 16]
        return output_mean, output_var

def generator1(noise, encode, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        # shape: [None, 4, 16]
        noise = tf.transpose(tf.concat([noise, encode], axis=2), perm=[0, 2, 1])
        # shape: [None, NOISE_LENGTH * 4 + 16, 4]
        noise = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, NOISE_LENGTH * 4 + 16, 4, 1, 1]
        deconv1 = conv(inputs=noise, filters=NOISE_LENGTH * 64, kernel_size=[1, 1, 2], strides=(1, 1, 2), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv1')
        # shape: [None, NOISE_LENGTH * 64, 4, 1, 2]
        deconv2 = conv(inputs=deconv1, filters=NOISE_LENGTH * 32, kernel_size=[1, 1, 3], strides=(1, 1, 3), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv2')
        # shape: [None, NOISE_LENGTH * 32, 4, 1, 6]
        deconv3 = conv(inputs=deconv2, filters=NOISE_LENGTH * 16, kernel_size=[1, 1, 4], strides=(1, 1, 4), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv3')
        # shape: [None, NOISE_LENGTH * 16, 4, 1, 24]
        deconv4 = conv(inputs=deconv3, filters=NOISE_LENGTH * 8, kernel_size=[1, 2, 1], strides=(1, 2, 1), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv4')
        # shape: [None, NOISE_LENGTH * 8, 4, 2, 24]
        deconv5 = conv(inputs=deconv4, filters=NOISE_LENGTH * 4, kernel_size=[1, 3, 1], strides=(1, 3, 1), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv5')
        # shape: [None, NOISE_LENGTH * 4, 4, 6, 24]
        deconv6 = conv(inputs=deconv5, filters=NOISE_LENGTH * 2, kernel_size=[1, 3, 1], strides=(1, 3, 1), \
                            training=train, regularization='batch_norm_relu', transpose=True, name='deconv6')
        # shape: [None, NOISE_LENGTH * 2, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv1 = conv(inputs=deconv6, filters=1, training=train, regularization='batch_norm_tanh', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        output = tf.transpose(conv1, perm=[0, 2, 3, 4, 1])
        # shape: [None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16, 1]
        output_image = tf.unstack(output[:BATCH_SIZE // 10], axis=1)
        # shape: [4, BATCH_SIZE // 10, CLASS_NUM // 4, INPUT_LENGTH // 16, 1]
        output_image = tf.concat(output_image, axis=2)
        # shape: [BATCH_SIZE // 10, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        tf.summary.image(name='piano_roll', tensor=output_image)
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        return output, deconv6

def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.split(axis=-1, value=encode, num_or_size_splits=8, name='encode_split')
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1, name='encode_stack')
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 10, 1, 9, 3))
        # shape: [None, 60, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]]
        inputs = tf.concat([inputs, encode], axis=1)
        # shape: [None, 124, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        res1 = residual_block(inputs=inputs, filters=64, training=train, regularization='batch_norm_relu', name='res1')
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        res2 = residual_block(inputs=res1, filters=64, training=train, regularization='batch_norm_relu', name='res2')
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        deconv1 = conv(inputs=res2, filters=32, kernel_size=[1, 2, 1], strides=(1, 2, 1), training=train, \
                                            regularization='batch_norm_relu', transpose=True, name='deconv1')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 16]
        deconv2 = conv(inputs=deconv1, filters=32, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, \
                                            regularization='batch_norm_relu', transpose=True, name='deconv2')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv1 = conv(inputs=deconv2, filters=1, training=train, regularization='batch_norm_tanh', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        output = tf.transpose(conv1, perm=[0, 2, 3, 4, 1])
        # shape: [None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8, 1]
        output_image = tf.unstack(output[:BATCH_SIZE // 10], axis=1)
        # shape: [4, BATCH_SIZE // 10, CLASS_NUM // 2, INPUT_LENGTH // 8, 1]
        output_image = tf.concat(output_image, axis=2)
        # shape: [BATCH_SIZE // 10, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        tf.summary.image(name='piano_roll', tensor=output_image)
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        return output, deconv2

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        encode = tf.split(axis=-1, value=encode, num_or_size_splits=8, name='encode_split')
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1, name='encode_stack')
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 5, 1, 18, 6), name='encode_tile')
        # shape: [None, 30, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        inputs = tf.concat([inputs, encode], axis=1)
        # shape: [None, 62, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        res1 = residual_block(inputs=inputs, filters=32, training=train, regularization='batch_norm_relu', name='res1')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        res2 = residual_block(inputs=res1, filters=32, training=train, regularization='batch_norm_relu', name='res2')
        # shape: [None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        deconv1 = conv(inputs=res2, filters=16, kernel_size=[1, 2, 1], strides=(1, 2, 1), training=train, \
                                            regularization='batch_norm_relu', transpose=True, name='deconv1')
        # shape: [None, 16, 4, CLASS_NUM, INPUT_LENGTH // 8]
        print(deconv1.get_shape())
        deconv2 = conv(inputs=deconv1, filters=16, kernel_size=[1, 1, 2], strides=(1, 1, 2), training=train, \
                                            regularization='batch_norm_relu', transpose=True, name='deconv2')
        # shape: [None, 16, 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = conv(inputs=deconv2, filters=1, training=train, regularization='batch_norm_tanh', name='conv1')
        # shape: [None, 1, 4, CLASS_NUM, INPUT_LENGTH // 4]
        output = tf.transpose(conv1, perm=[0, 2, 3, 4, 1])
        # shape: [None, 4, CLASS_NUM, INPUT_LENGTH // 4, 1]
        output_image = tf.unstack(output[:BATCH_SIZE // 10], axis=1)
        # shape: [4, BATCH_SIZE // 10, CLASS_NUM // 2, INPUT_LENGTH // 8, 1]
        output_image = tf.concat(output_image, axis=2)
        # shape: [BATCH_SIZE // 10, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        tf.summary.image(name='piano_roll', tensor=output_image)
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, CLASS_NUM, INPUT_LENGTH // 4]
        return output, deconv2

def generator4(inputs, encode, num, train):
    with tf.variable_scope('Generator4_' + str(num)):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.split(axis=-1, value=encode, num_or_size_splits=8, name='encode_split')
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1, name='encode_stack')
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(encode, multiples=(1, 3, 1, 36, 12))
        # shape: [None, 18, 4, CLASS_NUM, INPUT_LENGTH // 4]
        inputs = tf.concat([inputs, encode], axis=1)
        # shape: [None, 30, 4, CLASS_NUM, INPUT_LENGTH // 4]
        inputs = tf.unstack(inputs, axis=2, name='unstack')
        # shape: [4, None, 32, CLASS_NUM, INPUT_LENGTH // 4]
        inputs = tf.concat(inputs, axis=-1)
        # shape: [None, 32, CLASS_NUM, INPUT_LENGTH]
        res1 = residual_block(inputs=inputs, filters=16, training=train, regularization='batch_norm_relu', name='res1')
        # shape: [None, 16, CLASS_NUM, INPUT_LENGTH]
        res2 = residual_block(inputs=res1, filters=16, training=train, regularization='batch_norm_relu', name='res2')
        # shape: [None, 16, CLASS_NUM, INPUT_LENGTH]
        conv1 = conv(inputs=res2, filters=1, training=train, regularization='batch_norm_tanh', name='conv1')
        # shape: [None, 1, CLASS_NUM, INPUT_LENGTH]
        output = tf.transpose(conv1, perm=[0, 2, 3, 1], name='image')
        # shape: [None, CLASS_NUM, INPUT_LENGTH, 1]
        tf.summary.image(name='piano_roll', tensor=output[:BATCH_SIZE // 10])
        output = tf.squeeze(output, axis=-1, name='output')
        # shape: [None, CLASS_NUM, INPUT_LENGTH]
        return output

def discriminator1(inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv1 = conv(inputs=inputs, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv1')
        # shape: [None, 128, 2, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv2 = conv(inputs=conv1, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv2')
        # shape: [None, 128, 1, CLASS_NUM // 4, INPUT_LENGTH // 16]
        conv3 = conv(inputs=conv2, filters=128, kernel_size=[1, 3, 1], strides=(1, 3, 1), name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 12, INPUT_LENGTH // 16]
        conv4 = conv(inputs=conv3, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 16]
        conv5 = conv(inputs=conv4, filters=128, kernel_size=[1, 1, 2], strides=(1, 1, 2), name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 32]
        conv6 = conv(inputs=conv5, filters=256, kernel_size=[1, 1, 4], strides=(1, 1, 2), name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 64]
        conv7 = conv(inputs=conv6, filters=512, kernel_size=[1, 1, 3], strides=(1, 1, 2), name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, name='output')
        # shape: [None, 1]
        return output

def discriminator1_conditional(inputs, encode):
    with tf.variable_scope('Discriminator1_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.split(encode, num_or_size_splits=8, axis=-1)
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 9, 3))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        uncond = discriminator1(inputs=inputs, name='Discriminator1_Uncond')
        cond = discriminator1(inputs=tf.concat(values=[inputs, encode], axis=1), name='Discriminator1_Cond')
        output = (uncond + cond) / 2.0
        return output

def discriminator2(inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv1 = conv(inputs=inputs, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv1')
        # shape: [None, 128, 2, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv2 = conv(inputs=conv1, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv2')
        # shape: [None, 128, 1, CLASS_NUM // 2, INPUT_LENGTH // 8]
        conv3 = conv(inputs=conv2, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 12, INPUT_LENGTH // 8]
        conv4 = conv(inputs=conv3, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 8]
        conv5 = conv(inputs=conv4, filters=128, kernel_size=[1, 1, 2], strides=(1, 1, 2), name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 16]
        conv6 = conv(inputs=conv5, filters=256, kernel_size=[1, 1, 3], strides=(1, 1, 2), name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 32]
        conv7 = conv(inputs=conv6, filters=512, kernel_size=[1, 1, 4], strides=(1, 1, 4), name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, name='output')
        # shape: [None, 1]
        return output

def discriminator2_conditional(inputs, encode):
    with tf.variable_scope('Discriminator2_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.split(encode, num_or_size_splits=8, axis=-1)
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 18, 6))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        uncond = discriminator2(inputs=inputs, name='Discriminator2_Uncond')
        cond = discriminator2(inputs=tf.concat(values=[inputs, encode], axis=1), name='Discriminator2_Cond')
        output = (uncond + cond) / 2.0
        return output

def discriminator3(inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), 4, CLASS_NUM, INPUT_LENGTH // 4]
        conv1 = conv(inputs=inputs, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv1')
        # shape: [None, 128, 2, CLASS_NUM, INPUT_LENGTH // 4]
        conv2 = conv(inputs=conv1, filters=128, kernel_size=[2, 1, 1], strides=(2, 1, 1), name='conv2')
        # shape: [None, 128, 1, CLASS_NUM, INPUT_LENGTH // 4]
        conv3 = conv(inputs=conv2, filters=128, kernel_size=[1, 6, 1], strides=(1, 6, 1), name='conv3')
        # shape: [None, 128, 1, CLASS_NUM // 6, INPUT_LENGTH // 4]
        conv4 = conv(inputs=conv3, filters=128, kernel_size=[1, 12, 1], strides=(1, 12, 1), name='conv4')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 4]
        conv5 = conv(inputs=conv4, filters=128, kernel_size=[1, 1, 2], strides=(1, 1, 2), name='conv5')
        # shape: [None, 128, 1, 1, INPUT_LENGTH // 8]
        conv6 = conv(inputs=conv5, filters=256, kernel_size=[1, 1, 4], strides=(1, 1, 4), name='conv6')
        # shape: [None, 256, 1, 1, INPUT_LENGTH // 32]
        conv7 = conv(inputs=conv6, filters=512, kernel_size=[1, 1, 4], strides=(1, 1, 4), name='conv7')
        # shape: [None, 512, 1, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, name='output')
        # shape: [None, 1]
        return output


def discriminator3_conditional(inputs, encode):
    with tf.variable_scope('Discriminator3_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.split(encode, num_or_size_splits=8, axis=-1)
        # shape: [8, None, CHANNEL_NUM, 4, 2]
        encode = tf.stack(encode, axis=-1)
        # shape: [None, CHANNEL_NUM, 4, 2, 8]
        encode = tf.tile(input=encode, multiples=(1, 1, 1, 36, 12))
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        uncond = discriminator3(inputs=inputs, name='Discriminator3_Uncond')
        cond = discriminator3(inputs=tf.concat(values=[inputs, encode], axis=1), name='Discriminator3_Cond')
        output = (uncond + cond) / 2.0
        return output
    
def discriminator4(inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM * (2 or 1), CLASS_NUM, INPUT_LENGTH]
        conv1 = conv(inputs=inputs, filters=128, kernel_size=[3, 1], strides=(3, 1), name='conv1')
        # shape: [None, 128, CLASS_NUM // 3, INPUT_LENGTH]
        conv2 = conv(inputs=conv1, filters=128, kernel_size=[4, 1], strides=(4, 1), name='conv2')
        # shape: [None, 128, CLASS_NUM // 12, INPUT_LENGTH]
        conv3 = conv(inputs=conv2, filters=128, kernel_size=[6, 1], strides=(6, 1), name='conv3')
        # shape: [None, 128, 1, INPUT_LENGTH]
        conv4 = conv(inputs=conv3, filters=128, kernel_size=[1, 2], strides=(1, 2), name='conv4')
        # shape: [None, 6128, 1, INPUT_LENGTH // 2]
        conv5 = conv(inputs=conv4, filters=128, kernel_size=[1, 4], strides=(1, 4), name='conv5')
        # shape: [None, 128, 1, INPUT_LENGTH // 8]
        conv6 = conv(inputs=conv5, filters=256, kernel_size=[1, 4], strides=(1, 4), name='conv6')
        # shape: [None, 256, 1, INPUT_LENGTH // 32]
        conv7 = conv(inputs=conv6, filters=512, kernel_size=[1, 4], strides=(1, 4), name='conv7')
        # shape: [None, 512, 1, INPUT_LENGTH // 128]
        dense1 = tf.layers.dense(inputs=tf.layers.flatten(inputs=conv7), units=1024, activation=tf.nn.leaky_relu, name='dense1')
        # shape: [None, 1024]
        output = tf.layers.dense(inputs=dense1, units=1, name='output')
        # shape: [None, 1]
        return output

def discriminator4_conditional(inputs, encode):
    with tf.variable_scope('Discriminator4_Conditional', reuse=tf.AUTO_REUSE):
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode = tf.tile(input=encode, multiples=(1, 1, CLASS_NUM // 4, INPUT_LENGTH // 16))
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        uncond = discriminator4(inputs=inputs, name='Discriminator4_Uncond')
        cond = discriminator4(inputs=tf.concat(values=[inputs, encode], axis=1), name='Discriminator4_Cond')
        output = (uncond + cond) / 2.0
        return output

def get_noise(size):
    return normal(loc=0.0, scale=1.0, size=size)
