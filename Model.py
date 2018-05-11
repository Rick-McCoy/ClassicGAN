from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal

NOISE_LENGTH = 128

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv(inputs, filters, kernel_size=[1, 3, 3], strides=1, \
        training=True, regularization='lrelu', transpose=False, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if 'lrelu' in regularization:
            activation_function = tf.nn.leaky_relu
        elif 'relu' in regularization:
            activation_function = tf.nn.relu
        use_bias = not 'batch_norm' in regularization and not 'no_bias' in regularization
        if inputs.get_shape().ndims == 4:
            if type(kernel_size) is list and len(kernel_size) == 3:
                kernel_size = 3
            if transpose:
                conv_func = tf.layers.conv2d_transpose
            else:
                conv_func = tf.layers.conv2d
        elif transpose:
            conv_func = tf.layers.conv3d_transpose
        else:
            conv_func = tf.layers.conv3d
        if 'batch_norm' in regularization:
            output = tf.layers.batch_normalization(inputs=inputs, \
                                    axis=1, training=training, \
                                    name='batch_norm', fused=True)
            output = activation_function(output)
            output = conv_func(inputs=output, filters=filters, \
                            kernel_size=kernel_size, strides=strides, \
                            padding='same', data_format='channels_first', \
                            use_bias=False, name='conv')
        else:
            use_bias = not 'no_bias' in regularization
            output = conv_func(inputs=output, filters=filters, \
                            kernel_size=kernel_size, strides=strides, \
                            padding='same', data_format='channels_first', \
                            activation=activation_function, \
                            use_bias = use_bias, name='conv')

        return output

def residual_block(inputs, filters, training, strides=1, name=''):
    with tf.variable_scope(name):
        conv1 = conv(inputs=inputs, filters=filters, kernel_size=1, \
                    strides=1, training=training, \
                    regularization='batch_norm_relu', name='conv1')
        conv2 = conv(inputs=conv1, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv2')
        conv3 = conv(inputs=conv2, filters=filters, kernel_size=1, \
                    strides=1, training=training, \
                    regularization='batch_norm_relu', name='conv3')
        if inputs.get_shape().as_list()[1] != filters:
            skip = conv(inputs=inputs, filters=filters, \
            training=training, regularization='no_bias_relu', name='skip')
        else:
            skip = inputs
        return skip + conv3

def encode_concat(inputs, encode, name='encode_concat'):
    with tf.variable_scope(name):
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
        output = conv(output, filters=filters, training=training, \
                                regularization='batch_norm_relu', \
                                name='conv2')
        return output

def summary_image(inputs, name='summary_image'):
    with tf.variable_scope(name):
        output_image = tf.unstack(inputs, axis=1)
        output_image = tf.concat(output_image, axis=2)
        tf.summary.image(name='piano_roll', tensor=output_image)

def encoder(inputs, train):
    with tf.variable_scope('Encoder'):
        kernel_size = [[1, 1, 2], [1, 2, 2], [1, 2, 2], \
                    [1, 2, 2], [1, 3, 2], [1, 3, 3]]
        output = inputs
        for i, kernel in enumerate(kernel_size):
            output = conv(inputs=output, filters=16, \
            kernel_size=kernel, strides=tuple(kernel), \
            training=train, regularization='batch_norm_lrelu', \
            name='conv%d' % (i + 1))
        # shape: [None, 16, 4, 1, 1]
        output = tf.squeeze(output)
        output = tf.transpose(output, perm=[0, 2, 1])
        # shape: [None, 4, 16]
        output = tf.split(output, axis=-1, num_or_size_splits=2, \
                            name='output_split')
        # shape: [None, 4, 2, 8]
        output = tf.stack(output, axis=2)
        output = tf.expand_dims(output, axis=1)
        # shape: [None, 1, 4, 2, 8]
        return output

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
        output = tf.tanh(conv1)
        output = tf.transpose(output, perm=[0, 2, 3, 4, 1])
        summary_image(output[:1])
        output = tf.squeeze(output, axis=-1)
        return output, upblock1

def shared_gen(noise, encode, train):
    with tf.variable_scope('Shared_generator'):
        kernel_size = [[1, 1, 2], [1, 2, 2], [1, 3, 2], [1, 3, 3]]
        encode = tf.squeeze(encode, axis=1)
        encode = tf.unstack(encode, axis=-1)
        encode = tf.concat(encode, axis=-1)
        encode = tf.transpose(encode, perm=[0, 2, 1])
        noise = tf.concat([noise, encode], axis=1)
        output = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, 144, 4, 1, 1]
        for i, kernel in enumerate(kernel_size):
            output = conv(inputs=output, filters=1024 // 2 ** i, \
                            kernel_size=kernel, strides=tuple(kernel), \
                            training=train, regularization='batch_norm_relu', \
                            transpose=True, name='conv%d' % (i + 1))
        # shape: [None, 128, 4, 18, 24]
        output = conv(inputs=output, filters=64, training=train, \
                        regularization='batch_norm_relu', name='conv5')
        output = tf.tanh(output)
        return output

def generator1(inputs, encode, num, train):
    with tf.variable_scope('Generator1_%d' % num):
        return genblock(inputs, encode, 128, train)

def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_%d' % num):
        return genblock(inputs, encode, 64, train)

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_%d' % num):
        # shape: [None, 4, 16]
        inputs = encode_concat(inputs, encode)
        # shape: [None, 33, 4, 72, 96]
        inputs = tf.unstack(inputs, axis=2, name='unstack')
        # shape: [4, None, 33, 72, 96]
        inputs = tf.concat(inputs, axis=-1)
        # shape: [None, 33, 72, 384]
        res1 = residual_block(inputs=inputs, filters=16, \
                            training=train, name='res1')
        # shape: [None, 16, 72, 384]
        res2 = residual_block(inputs=res1, filters=16, \
                            training=train, name='res2')
        # shape: [None, 16, 72, 384]
        conv1 = conv(inputs=res2, filters=1, training=train, \
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
        if inputs.get_shape().ndims == 5:
            kernel_size=[[2, 1, 1], [2, 1, 1], [1, 2, 2], [1, 3, 3]]
            strides = (1, 2, 2)
        else:
            kernel_size=[[1, 2], [1, 2], [2, 2], [3, 3]]
            strides = 2
        output = inputs
        for i, kernel in enumerate(kernel_size):
            output = conv(inputs=output, filters=filters, \
                        kernel_size=kernel, strides=tuple(kernel), \
                        name='conv%d' % (i + 1))
            filters = filters * 2
        num = 5
        while output.get_shape().as_list()[-2] > 3:
            output = conv(inputs=output, filters=filters, \
                        kernel_size=kernel_size[3], strides=strides, \
                        name='conv%d' % num)
            filters = filters * 2
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
        # shape: [None, 6, 4, 36, 48]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 256, 1, 3, 4]
        block = block3x3(down, filters=256, name='block1')
        # shape: [None, 64, 1, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def discriminator2(inputs, encode):
    with tf.variable_scope('Discriminator2', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 4, 72, 96]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 512, 1, 3, 4]
        block = block3x3(down, filters=512, name='block1')
        # shape: [None, 128, 1, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def discriminator3(inputs, encode):
    with tf.variable_scope('Discriminator3', reuse=tf.AUTO_REUSE):
        # shape: [None, 6, 72, 384]
        down = downsample(inputs, filters=16, name='downsample')
        # shape: [None, 512, 3, 4]
        block = block3x3(down, filters=512, name='block1')
        # shape: [None, 128, 3, 4]
        return conditional_output(block, encode, 'cond_out')

def get_noise(size):
    return normal(loc=0.0, scale=1.0, size=size)
