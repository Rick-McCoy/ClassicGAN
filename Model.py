from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal

NOISE_LENGTH = 128

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv(inputs, filters, kernel_size=3, strides=1, \
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
        if transpose:
            conv_func = tf.layers.conv2d_transpose
        else:
            conv_func = tf.layers.conv2d
        use_bias = not 'batch_norm' in regularization and not 'no_bias' in regularization
        output = conv_func(inputs=inputs, filters=filters, \
                        kernel_size=kernel_size, strides=strides, \
                        padding='same', data_format='channels_first', \
                        activation=activation, \
                        use_bias = use_bias, name='conv')
        if 'batch_norm' in regularization:
            output = tf.layers.batch_normalization(inputs=output, \
                                    axis=1, training=training, \
                                    name='batch_norm')
        return output

def encoder(inputs, train):
    with tf.variable_scope('Encoder'):
        output = inputs
        i = 0
        while output.get_shape().as_list()[-1] > 1:
            output = conv(inputs=output, filters=2 ** (i // 2 + 3), \
                strides=2, training=train, \
                regularization='batch_norm_lrelu', \
                name='conv%d' % (i + 1))
            i += 1
        output = tf.squeeze(output)
        output = tf.layers.dense(inputs=output, units=64, name='dense1')
        # shape: [None, 64]
        return output

def upsample(inputs, filters, training, name='upsample'):
    with tf.variable_scope(name):
        size = [i * 2 for i in inputs.get_shape().as_list()[-2:]]
        output = tf.transpose(inputs, [0, 2, 3, 1])
        output = tf.image.resize_bilinear(output, size)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = conv(output, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv1')
        output = conv(output, filters=filters, training=training, \
                    regularization='batch_norm_relu', name='conv2')
        return output

def encode_concat(inputs, encode, name='encode_concat'):
    with tf.variable_scope(name):
        encode = tf.reshape(encode, [-1, 1, 4, 16])
        dim1 = inputs.get_shape().as_list()[-2] // 4
        dim2 = inputs.get_shape().as_list()[-1] // 16
        encode = tf.tile(input=encode, multiples=(1, 1, dim1, dim2))
        output = tf.concat([inputs, encode], axis=1)
        return output

def genblock(inputs, encode, filters, train, name='genblock'):
    with tf.variable_scope(name):
        inputs = encode_concat(inputs, encode)
        upsample1 = upsample(inputs, filters=filters, training=train)
        conv1 = conv(inputs=upsample1, filters=1, training=train, \
                    regularization='tanh', name='conv1')
        output = tf.transpose(conv1, perm=[0, 2, 3, 1])
        tf.summary.image(name='piano_roll', tensor=output[:1])
        output = tf.squeeze(output, axis=-1)
        return output, upsample1

def shared_gen(noise, encode, train):
    with tf.variable_scope('Shared_generator'):
        noise = tf.concat([noise, encode], axis=1)
        # shape: [None, 192]
        output = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, 192, 1, 1]
        for i in range(4):
            output = upsample(output, filters=1024 // 2 ** i, \
                            training=train, name='gneblock%d' % (i + 1))
        # shape: [None, 128, 16, 16]
        output = conv(inputs=output, filters=64, strides=(1, 2), \
                    training=train, regularization='batch_norm_relu', \
                    transpose=True, name='conv5')
        output = conv(inputs=output, filters=64, strides=(1, 2), \
                    training=train, regularization='batch_norm_relu', \
                    transpose=True, name='conv6')
        output = conv(inputs=output, filters=64, training=train, \
                        regularization='tanh', name='conv7')
        return output

def generator1(inputs, encode, num, train):
    with tf.variable_scope('Generator1_%d' % num):
        return genblock(inputs, encode, 64, train)

def generator2(inputs, encode, num, train):
    with tf.variable_scope('Generator2_%d' % num):
        return genblock(inputs, encode, 32, train)

def generator3(inputs, encode, num, train):
    with tf.variable_scope('Generator3_%d' % num):
        output, _ = genblock(inputs, encode, 16, train)
        return output

def downblock(inputs, filters, name='downblock'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = tf.layers.average_pooling2d(inputs, pool_size=2, strides=2, \
                                    padding='same', data_format='channels_first')
        output = conv(output, filters=filters, name='conv1')
        output = conv(output, filters=filters, name='conv2')
        return output

def downsample(inputs, name='downsample'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filters = 16
        output = inputs
        i = 1
        while output.get_shape().as_list()[-2] > 1:
            output = downblock(output, filters=filters * (2 ** i), name='downblock%d' % i)
            i += 1
        return output

def conditional_output(inputs, encode, name='cond_out'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output1 = tf.layers.flatten(inputs)
        output2 = tf.concat([output1, encode], axis=1)
        output1 = tf.layers.dense(output1, units=1, name='output1')
        output2 = tf.layers.dense(output2, units=1, name='output2')
        return (output1 + output2) / 2
    
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
