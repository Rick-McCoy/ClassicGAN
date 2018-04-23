from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal

NOISE_LENGTH = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv(inputs, filters, kernel_size=[1, 3, 3], strides=(1, 1, 1), \
        training=True, regularization='lrelu', transpose=False, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if regularization[-5:] == 'lrelu':
            activation_function = tf.nn.leaky_relu
        elif regularization[-4:] == 'relu':
            activation_function = tf.nn.relu
        use_bias = regularization[:10] != 'batch_norm'
        if not use_bias:
            output = tf.layers.batch_normalization(inputs=inputs, \
                                        axis=1, training=training, \
                                        name='batch_norm', fused=True)
        else:
            output = inputs
        if inputs.get_shape().ndims == 4:
            if len(kernel_size) == 3:
                kernel_size = [3, 3]
                strides = (1, 1)
            if transpose:
                conv_func = tf.layers.conv2d_transpose
            else:
                conv_func = tf.layers.conv2d
        elif transpose:
            conv_func = tf.layers.conv3d_transpose
        else:
            conv_func = tf.layers.conv3d
        output = activation_function(output)
        output = conv_func(inputs=output, filters=filters, \
                        kernel_size=kernel_size, strides=strides, \
                        padding='same', data_format='channels_first', \
                        use_bias=use_bias, name='conv')
        return output

def residual_block(inputs, filters, training, name=''):
    with tf.variable_scope(name):
        if inputs.get_shape().as_list()[1] != filters:
            inputs = conv(inputs=inputs, filters=filters, \
            training=training, regularization='relu', name='inputs')
        conv1 = conv(inputs=inputs, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv1')
        conv2 = conv(inputs=conv1, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv2')
        return inputs + conv2

def encode_concat(inputs, encode, name='encode_concat'):
    with tf.variable_scope(name):
        encode = tf.split(encode, axis=-1, num_or_size_splits=8, \
                            name='encode_split')
        encode = tf.stack(encode, axis=-1, name='encode_stack')
        encode = tf.split(encode, axis=3, num_or_size_splits=4, \
                            name='encode_split_2')
        encode = tf.concat(encode, axis=1, name='encode_concat')
        dim1 = inputs.get_shape().as_list()[3] // 2
        dim2 = inputs.get_shape().as_list()[4] // 8
        encode = tf.tile(input=encode, multiples=(1, 1, 1, dim1, dim2))
        output = tf.concat([inputs, encode], axis=1)
        return output

def upblock(inputs, filters, training, name='upblock'):
    with tf.variable_scope(name):
        output = conv(inputs, filters=filters, kernel_size=[1, 2, 2], \
                                strides=(1, 2, 2), training=training, \
                                regularization='batch_norm_relu', \
                                transpose=True, name='conv1')
        output = conv(output, filters=filters, kernel_size=[1, 3, 3], \
                                strides=(1, 1, 1), training=training, \
                                regularization='batch_norm_relu', \
                                transpose=True, name='conv2')
        return output

def summary_image(inputs, name='summary_image'):
    with tf.variable_scope(name):
        output_image = tf.unstack(inputs, axis=1)
        output_image = tf.concat(output_image, axis=2)
        tf.summary.image(name='piano_roll', tensor=output_image)

def noise_generator(noise, train):
    with tf.variable_scope('Noise_generator'):
        # shape: [None, 1, 1, 32]
        conv1 = conv(inputs=noise, filters=1024, kernel_size=[1, 3], \
                    strides=(1, 1), training=train, \
                    regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1024, 1, 32]
        conv2 = conv(inputs=conv1, filters=4, kernel_size=[1, 3], \
                    strides=(1, 1), training=train, \
                    regularization='batch_norm_relu', name='conv2')
        # shape: [None, 4, 1, 32]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, 32]
        return output

def time_seq_noise_generator(noise, num, train):
    with tf.variable_scope('Time_seq_noise_generator' + str(num)):
        # shape: [None, 1, 1, 32]
        conv1 = conv(inputs=noise, filters=1024, kernel_size=[1, 3], \
                    strides=(1, 1), training=train, \
                    regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1024, 1, 32]
        conv2 = conv(inputs=conv1, filters=4, kernel_size=[1, 3], \
                    strides=(1, 1), training=train, \
                    regularization='batch_norm_relu', name='conv2')
        # shape: [None, 4, 1, 32]
        output = tf.transpose(conv2, perm=[0, 2, 1, 3], name='output')
        # shape: [None, 1, 4, 32]
        return output

def encoder(inputs, num, train):
    with tf.variable_scope('Encoder' + str(num)):
        kernel_size = [[1, 1, 2], [1, 2, 2], [1, 2, 2], \
                    [1, 2, 2], [1, 3, 2], [1, 3, 3]]
        output = inputs
        for i, kernel in enumerate(kernel_size):
            output = conv(inputs=output, filters=2 ** (i + 1), \
            kernel_size=kernel, strides=tuple(kernel), \
            training=train, regularization='batch_norm_lrelu', \
            name='conv' + str(i + 1))
        # shape: [None, 64, 4, 1, 1]
        output = tf.stack([tf.layers.flatten(i) \
                        for i in tf.unstack(output, axis=2)], axis=1)
        # shape: [None, 4, 64]
        return output

def generator1(noise, encode, num, train):
    with tf.variable_scope('Generator1_' + str(num)):
        noise = tf.transpose(tf.concat([noise, encode], axis=2), \
                            perm=[0, 2, 1])
        # shape: [None, 192, 4]
        noise = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, 192, 4, 1, 1]
        kernel_size = [[1, 1, 2], [1, 2, 2], [1, 3, 2], [1, 3, 3]]
        transconv = noise
        for i, kernel in enumerate(kernel_size):
            transconv = conv(inputs=transconv, filters=1024 // 2 ** i, \
                    kernel_size=kernel, strides=tuple(kernel), \
                    training=train, regularization='batch_norm_relu', \
                    transpose=True, name='transconv' + str(i + 1))
        # shape: [None, 128, 4, 18, 24]
        conv1 = conv(inputs=transconv, filters=1, training=train, \
                    regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1, 4, 18, 24]
        output = tf.transpose(tf.tanh(conv1), perm=[0, 2, 3, 4, 1])
        # shape: [None, 4, 18, 24, 1]
        summary_image(output[:1])
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 4, 18, 24]
        return output, transconv

def genblock(inputs, encode, filters, train, name='genblock'):
    with tf.variable_scope(name):
        inputs = encode_concat(inputs, encode)
        res1 = residual_block(inputs=inputs, filters=filters, \
                            training=train, name='res1')
        res2 = residual_block(inputs=res1, filters=filters, \
                            training=train, name='res2')
        upblock1 = upblock(res2, filters=filters // 2, training=train)
        conv1 = conv(inputs=upblock1, filters=1, training=train, \
                    regularization='batch_norm_relu', name='conv1')
        output = tf.transpose(tf.tanh(conv1), perm=[0, 2, 3, 4, 1])
        summary_image(output[:1])
        output = tf.squeeze(output, axis=-1)
        return output, upblock1


def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_' + str(num)):
        return genblock(inputs, encode, 64, train)

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_' + str(num)):
        return genblock(inputs, encode, 32, train)

def generator4(inputs, encode, num, train):
    with tf.variable_scope('Generator4_' + str(num)):
        # shape: [None, 6, 4, 64]
        inputs = encode_concat(inputs, encode)
        # shape: [None, 46, 4, 72, 96]
        inputs = tf.unstack(inputs, axis=2, name='unstack')
        # shape: [4, None, 46, 72, 96]
        inputs = tf.concat(inputs, axis=-1)
        # shape: [None, 46, 72, 384]
        res1 = residual_block(inputs=inputs, filters=16, \
                            training=train, name='res1')
        # shape: [None, 16, 72, 384]
        res2 = residual_block(inputs=res1, filters=16, \
                            training=train, name='res2')
        # shape: [None, 16, 72, 384]
        res3 = residual_block(inputs=res2, filters=16, \
                            training=train, name='res3')
        # shape: [None, 16, 72, 384]
        conv1 = conv(inputs=res3, filters=1, training=train, \
                    regularization='batch_norm_relu', name='conv1')
        # shape: [None, 1, 72, 384]
        output = tf.tanh(conv1)
        # shape: [None, 1, 72, 384]
        output = tf.transpose(output, perm=[0, 2, 3, 1], name='image')
        # shape: [None, 72, 384, 1]
        tf.summary.image(name='piano_roll', tensor=output[:1])
        output = tf.squeeze(output, axis=-1, name='output')
        # shape: [None, 72, 384]
        return output

def downsample(inputs, filters, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = conv(inputs=inputs, filters=filters, \
                    kernel_size=[2, 1, 1], strides=(2, 1, 1), \
                    name='conv1')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[2, 1, 1], strides=(2, 1, 1), \
                    name='conv2')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[1, 2, 2], strides=(1, 2, 2), \
                    name='conv3')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[1, 3, 3], strides=(1, 3, 3), \
                    name='conv4')
        num = 5
        while output.get_shape().as_list()[3] > 3:
            filters = filters * 2
            output = conv(inputs=output, filters=filters, \
                        kernel_size=[1, 3, 3], strides=(1, 2, 2), \
                        name='conv' + str(num))
            num += 1
        return output

def downsample2d(inputs, filters, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = conv(inputs=inputs, filters=filters, \
                    kernel_size=[1, 2], strides=(1, 2), name='conv1')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[1, 2], strides=(1, 2), name='conv2')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[2, 2], strides=(2, 2), name='conv3')
        filters = filters * 2
        output = conv(inputs=output, filters=filters, \
                    kernel_size=[3, 3], strides=(3, 3), name='conv4')
        num = 5
        while output.get_shape().as_list()[2] > 3:
            filters = filters * 2
            output = conv(inputs=output, filters=filters, \
                    kernel_size=[2, 2], strides=(2, 2), \
                    name='conv' + str(num))
            num += 1
        return output

def block3x3(inputs, filters, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filters = filters / 2
        output = conv(inputs=inputs, filters=filters, name='conv1')
        filters = filters / 2
        output = conv(inputs=output, filters=filters, name='conv2')
        return output

def conditional_output(inputs, encode, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        flatten = tf.layers.flatten(inputs=inputs)
        enc_flat = tf.layers.flatten(inputs=encode)
        join = tf.concat([flatten, enc_flat], axis=1)
        dense1 = tf.layers.dense(inputs=flatten, units=1024, \
                                activation=tf.nn.leaky_relu, \
                                name='dense1')
        dense2 = tf.layers.dense(inputs=join, units=1024, 
                                activation=tf.nn.leaky_relu, \
                                name='dense2')
        output1 = tf.layers.dense(inputs=dense1, units=1, \
                                name='output1')
        output2 = tf.layers.dense(inputs=dense2, units=1, \
                                name='output2')
        return (output1 + output2) / 2

def discriminator1(inputs, encode):
    with tf.variable_scope('Discriminator1', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 4, 18, 24]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 128, 1, 3, 4]
        block = block3x3(down, filters=128, name='block1')
        # shape: [None, 32, 1, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def discriminator2(inputs, encode):
    with tf.variable_scope('Discriminator2', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 4, 36, 48]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 256, 1, 3, 4]
        block = block3x3(down, filters=256, name='block1')
        # shape: [None, 64, 1, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def discriminator3(inputs, encode):
    with tf.variable_scope('Discriminator3', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 4, 72, 96]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 512, 1, 3, 4]
        block = block3x3(down, filters=512, name='block1')
        # shape: [None, 128, 1, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def discriminator4(inputs, encode):
    with tf.variable_scope('Discriminator4', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 72, 384]
        down = downsample2d(inputs, filters=16, name='downsample2d')
        # shape: [None, 512, 3, 4]
        block = block3x3(down, filters=512, name='block1')
        # shape: [None, 128, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def get_noise(size):
    return normal(loc=0.0, scale=1.0, size=size)
