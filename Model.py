from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal

NOISE_LENGTH = 128

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv(inputs, filters, kernel_size=[1, 3, 3], strides=1, \
        training=True, regularization='lrelu', transpose=False, name='conv'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if 'lrelu' in regularization:
            activation = tf.nn.leaky_relu
        elif 'relu' in regularization:
            activation = tf.nn.relu
        elif 'tanh' in regularization:
            activation = tf.tanh
        else:
            activation = None
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
            inputs = tf.layers.batch_normalization(inputs=inputs, \
                                    axis=1, training=training, \
                                    name='batch_norm', fused=True)
            use_bias = False
        else:
            use_bias = not 'no_bias' in regularization
        output = conv_func(inputs=inputs, filters=filters, \
                        kernel_size=kernel_size, strides=strides, \
                        padding='same', data_format='channels_first', \
                        activation=activation, \
                        use_bias = use_bias, name='conv')

        return output

def residual_block(inputs, filters, training, name='resblock'):
    with tf.variable_scope(name):
        conv1 = conv(inputs=inputs, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv1')
        conv2 = conv(inputs=conv1, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv2')
        return inputs + conv2

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
        output = inputs
        i = 0
        while output.get_shape().as_list() > 1:
            output = conv(inputs=output, filters=16, \
                strides=(1, 3, 3), training=train, \
                regularization='batch_norm_lrelu', \
                name='conv%d' % (i + 1))
            i += 1
        output = conv(inputs=output, filters=16, training=train, \
            regularization='batch_norm', name='conv7')
        # shape: [None, 16, 4, 1, 1]
        output = tf.squeeze(output, axis=-1)
        # shape: [None, 16, 4, 1]
        output = tf.transpose(output, perm=[0, 3, 2, 1])
        # shape: [None, 1, 4, 16]
        output = tf.split(output, axis=-1, num_or_size_splits=2, \
                            name='output_split')
        # shape: [2, None, 1, 4, 8]
        output = tf.stack(output, axis=-2)
        # shape: [None, 1, 4, 2, 8]
        return output

def genblock(inputs, encode, filters, train, name='genblock'):
    with tf.variable_scope(name):
        inputs = encode_concat(inputs, encode)
        conv1 = conv(inputs, filters=filters, training=train, \
                        regularization='batch_norm_relu', name='conv1')
        res1 = residual_block(inputs=conv1, filters=filters, \
                            training=train, name='res1')
        res2 = residual_block(inputs=res1, filters=filters, \
                            training=train, name='res2')
        upblock1 = upblock(res2, filters=filters // 2, training=train)
        conv2 = conv(inputs=upblock1, filters=1, training=train, \
                    regularization='batch_norm_tanh', name='conv2')
        output = tf.transpose(conv2, perm=[0, 2, 3, 4, 1])
        summary_image(output[:1])
        output = tf.squeeze(output, axis=-1)
        return output, upblock1

def shared_gen(noise, encode, train):
    with tf.variable_scope('Shared_generator'):
        kernel_size = [[1, 1, 2], [1, 2, 2], [1, 3, 2], [1, 3, 3]]
        encode = tf.reshape(tf.squeeze(encode), shape=[-1, 4, 16])
        # shape: [None, 4, 16]
        encode = tf.transpose(encode, perm=[0, 2, 1])
        # shape: [None, 16, 4]
        noise = tf.concat([noise, encode], axis=1)
        # shape: [None, 144, 4]
        output = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, 144, 4, 1, 1]
        for i, kernel in enumerate(kernel_size):
            output = conv(inputs=output, filters=1024 // 2 ** i, \
                            strides=tuple(kernel), training=train, \
                            regularization='batch_norm_relu', \
                            transpose=True, name='conv%d' % (2 * i + 1))
            output = conv(inputs=output, filters=1024 // 2 ** i, \
                            training=train, regularization='batch_norm_relu', \
                            name='conv%d' % (2 * i + 2))
        # shape: [None, 128, 4, 18, 24]
        output = conv(inputs=output, filters=64, training=train, \
                        regularization='batch_norm_tanh', name='conv9')
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
        conv1 = conv(inputs=inputs, filters=16, training=train, \
                    regularization='batch_norm_relu', name='conv1')
        # shape: [None, 16, 72, 384]
        res1 = residual_block(inputs=inputs, filters=16, \
                            training=train, name='res1')
        # shape: [None, 16, 72, 384]
        res2 = residual_block(inputs=res1, filters=16, \
                            training=train, name='res2')
        # shape: [None, 16, 72, 384]
        conv2 = conv(inputs=res2, filters=1, training=train, \
                    regularization='batch_norm_tanh', name='conv2')
        # shape: [None, 1, 72, 384]
        output = tf.transpose(conv2, perm=[0, 2, 3, 1], name='image')
        # shape: [None, 72, 384, 1]
        tf.summary.image(name='piano_roll', tensor=output[:1])
        output = tf.squeeze(output, axis=-1, name='output')
        # shape: [None, 72, 384]
        return output

def downblock(inputs, filters, name='downblock'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv(inputs, filters=filters, name='conv1')
        conv2 = conv(conv1, filters=filters, name='conv2')
        return inputs + conv2

def downsample(inputs, name='downsample'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filters = 16
        output = inputs
        i = 1
        while output.get_shape().as_list()[-2] > 3:
            if output.get_shape().as_list()[2] > 1 or output.get_shape().ndims == 4:
                kernel = 3
                strides = 2
            else:
                kernel = [1, 3, 3]
                strides = [1, 2, 2]
            output = conv(output, filters=filters * (2 ** i), \
                            kernel_size=kernel, strides=strides, \
                            name='conv%d' % i)
            output = downblock(output, filters=filters * (2 ** i), name='downblock%d' % i)
            i += 1
        for j in range(2):
            if output.get_shape().as_list()[-1] > 4:
                output = conv(inputs=output, filters=output.get_shape().as_list()[1] // 2, \
                                kernel_size=3, strides=(1, 2), \
                                name='conv%d' % (i + j))
            else:
                output = conv(inputs=output, filters=output.get_shape().as_list()[1] // 2, \
                                name='conv%d' % (i + j))
        return output

def conditional_output(inputs, encode, name='cond_out'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = inputs.get_shape().as_list()
        output1_shape = [shape[0], shape[1], 1]
        for i in shape[2:]:
            output1_shape[2] *= i
        output1 = tf.reshape(inputs, shape=output1_shape)
        encode_flat = tf.expand_dims(tf.layers.flatten(encode), axis=-1)
        encode_flat = tf.tile(encode_flat, multiples=(1, 1, output1.get_shape().as_list()[-1]))
        output2 = tf.concat([output1, encode_flat], axis=1)
        while output1.get_shape().as_list()[-1] > 1:
            output1 = tf.layers.conv1d(output1, filters=output1.get_shape().as_list()[1], \
                                        kernel_size=3, strides=2, padding='same', \
                                        data_format='channels_first', \
                                        activation=tf.nn.leaky_relu)
        output1 = tf.layers.conv1d(output1, filters=1, kernel_size=1, strides=1, \
                                    padding='same', data_format='channels_first')
        while output2.get_shape().as_list()[-1] > 1:
            output2 = tf.layers.conv1d(output2, filters=output2.get_shape().as_list()[1], \
                                        kernel_size=3, strides=2, padding='same', \
                                        data_format='channels_first', \
                                        activation=tf.nn.leaky_relu)
        output2 = tf.layers.conv1d(output2, filters=1, kernel_size=1, strides=1, \
                                    padding='same', data_format='channels_first')
        return (tf.layers.flatten(output1) + tf.layers.flatten(output2)) / 2
    
def discriminator1(inputs, encode, name='Discriminator1'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs)
        return conditional_output(inputs=down, encode=encode)
    
def discriminator2(inputs, encode, name='Discriminator2'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs)
        return conditional_output(inputs=down, encode=encode)
    
def discriminator3(inputs, encode, name='Discriminator3'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs)
        return conditional_output(inputs=down, encode=encode)

def get_noise(size):
    return normal(loc=0.0, scale=1.0, size=size)
