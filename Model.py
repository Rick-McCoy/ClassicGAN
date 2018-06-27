from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from numpy.random import normal

NOISE_LENGTH = 128
SPECTRAL_UPDATE_OPS = 'spectral_update_ops'
NO_OPS = 'no_ops'
ITERS = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _l2_normalize(inputs):
    return inputs / (tf.reduce_sum(inputs ** 2) ** 0.5 + 1e-12)

def spectral_norm(inputs, update_collection=None):
    input_shape = inputs.get_shape().as_list()
    w = tf.reshape(inputs, [-1, input_shape[-1]])
    u = tf.get_variable('u', shape=[1, input_shape[-1]], dtype=tf.float32, trainable=False)
    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2_normalize(tf.matmul(u_i, tf.transpose(w)))
        u_ip1 = _l2_normalize(tf.matmul(v_ip1, w))
        return i + 1, u_ip1, v_ip1
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < ITERS,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                u, tf.zeros(dtype=tf.float32, shape=[1, w.shape.as_list()[0]]))
    )
    w = w / tf.matmul(tf.matmul(v_final, w), tf.transpose(u_final))[0, 0]
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_final)]):
            w_norm = tf.reshape(w, input_shape)
    else:
        w_norm = tf.reshape(w, input_shape)
        if update_collection is not NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    return w_norm

def conv(inputs, channels, kernel_size_h=3, kernel_size_w=3, strides_h=1, strides_w=1, \
        update_collection=tf.GraphKeys.UPDATE_OPS, regularization='lrelu', transpose=False, name='conv'):
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
            filters = tf.get_variable(name='filters', shape=[kernel_size_h, kernel_size_w, \
                                    channels, inputs.get_shape().as_list()[1]], dtype=tf.float32)
            #filters = spectral_norm(filters, update_collection=update_collection)
            out_shape = inputs.get_shape().as_list()
            out_shape[1] = channels
            out_shape[2] *= strides_h
            out_shape[3] *= strides_w
            output = tf.nn.conv2d_transpose(inputs, filter=filters, output_shape=out_shape, \
                                        strides=[1, 1, strides_h, strides_w], padding='SAME', \
                                        data_format='NCHW', name='conv_trans')
        else:
            filters = tf.get_variable(name='filters', shape=[kernel_size_h, kernel_size_w, \
                                    inputs.get_shape().as_list()[1], channels], dtype=tf.float32)
            #filters = spectral_norm(filters, update_collection=update_collection)
            output = tf.nn.conv2d(input=inputs, filter=filters, strides=[1, 1, strides_h, strides_w], \
                                padding='SAME', data_format='NCHW', name='conv')
        if not 'no_bias' in regularization:
            bias = tf.get_variable(name='bias', shape=[channels], dtype=tf.float32, \
                                initializer=tf.zeros_initializer())
            output = tf.nn.bias_add(output, bias, data_format='NCHW')
        if activation is not None:
            output = activation(output)
        return output

def dense(inputs, units, update_collection, name='dense'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weight', shape=[inputs.get_shape().as_list()[1], units], \
                            dtype=tf.float32)
        #w = spectral_norm(w, update_collection=update_collection)
        b = tf.get_variable(name='bias', shape=[units], dtype=tf.float32, \
                            initializer=tf.zeros_initializer())
        return tf.nn.bias_add(tf.matmul(inputs, w), b)

def encoder(inputs, update_collection, train=True):
    with tf.variable_scope('Encoder'):
        output = inputs
        i = 0
        while output.get_shape().as_list()[-1] > 1:
            output = conv(inputs=output, channels=2 ** (i // 2 + 3), \
                strides_h=2, strides_w=2, update_collection=update_collection, \
                regularization='lrelu', \
                name='conv%d' % (i + 1))
            if i % 2 == 0:
                output = tf.layers.batch_normalization(output, axis=1, training=train)
            i += 1
        output = tf.squeeze(output)
        output = dense(inputs=output, units=64, \
                                update_collection=update_collection, name='dense1')
        # shape: [None, 64]
        return output

def upsample(inputs, channels, update_collection, name='upsample'):
    with tf.variable_scope(name):
        size = [i * 2 for i in inputs.get_shape().as_list()[-2:]]
        output = tf.transpose(inputs, [0, 2, 3, 1])
        output = tf.image.resize_bilinear(output, size)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = conv(output, channels=channels, update_collection=update_collection, \
                    regularization='relu', name='conv1')
        output = conv(output, channels=channels, update_collection=update_collection, \
                    regularization='relu', name='conv2')
        return output

def encode_concat(inputs, encode, name='encode_concat'):
    with tf.variable_scope(name):
        encode = tf.reshape(encode, [-1, 1, 4, 16])
        dim1 = inputs.get_shape().as_list()[-2] // 4
        dim2 = inputs.get_shape().as_list()[-1] // 16
        encode = tf.tile(input=encode, multiples=(1, 1, dim1, dim2))
        output = tf.concat([inputs, encode], axis=1)
        return output

def genblock(inputs, encode, channels, update_collection, train, name='genblock'):
    with tf.variable_scope(name):
        inputs = encode_concat(inputs, encode)
        inputs = tf.layers.batch_normalization(inputs, axis=1, training=train)
        upsample1 = upsample(inputs, channels=channels, update_collection=update_collection)
        upsample1 = tf.layers.batch_normalization(upsample1, axis=1, training=train)
        conv1 = conv(inputs=upsample1, channels=1, update_collection=update_collection, \
                    regularization='tanh', name='conv1')
        output = tf.transpose(conv1, perm=[0, 2, 3, 1])
        tf.summary.image(name='piano_roll', tensor=output[:1])
        output = tf.squeeze(output, axis=-1)
        return output, upsample1

def shared_gen(noise, encode, update_collection, train):
    with tf.variable_scope('Shared_generator'):
        noise = tf.concat([noise, encode], axis=1)
        # shape: [None, 192]
        output = tf.expand_dims(tf.expand_dims(noise, axis=-1), axis=-1)
        # shape: [None, 192, 1, 1]
        output = conv(inputs=output, channels=256, strides_h=1, strides_w=2, \
                    update_collection=update_collection, regularization='relu', \
                    transpose=True, name='conv1')
        output = conv(inputs=output, channels=256, strides_h=1, strides_w=2, \
                    update_collection=update_collection, regularization='relu', \
                    transpose=True, name='conv2')
        for i in range(4):
            output = upsample(output, channels=1024 // 2 ** i, \
                            update_collection=update_collection, name='genblock%d' % (i + 3))
            output = tf.layers.batch_normalization(output, axis=1, training=train)
        # shape: [None, 128, 16, 64]
        output = conv(inputs=output, channels=64, update_collection=update_collection, \
                        regularization='relu', name='conv7')
        return output

def generator1(inputs, encode, num, update_collection, train):
    with tf.variable_scope('Generator1_%d' % num):
        return genblock(inputs, encode, 64, update_collection, train)

def generator2(inputs, encode, num, update_collection, train):
    with tf.variable_scope('Generator2_%d' % num):
        return genblock(inputs, encode, 32, update_collection, train)

def generator3(inputs, encode, num, update_collection, train):
    with tf.variable_scope('Generator3_%d' % num):
        output, _ = genblock(inputs, encode, 16, update_collection, train)
        return output

def downblock(inputs, channels, update_collection, name='downblock'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = tf.layers.average_pooling2d(inputs, pool_size=2, \
                                    strides=2, padding='same', \
                                    data_format='channels_first')
        output = conv(output, channels=channels, update_collection=update_collection, name='conv1')
        output = conv(output, channels=channels, update_collection=update_collection, name='conv2')
        return output

def downsample(inputs, update_collection, name='downsample'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        channels = 16
        output = conv(inputs, channels=channels, update_collection=update_collection, name='conv1')
        output = conv(output, channels=channels, update_collection=update_collection, name='conv2')
        i = 1
        while output.get_shape().as_list()[-2] > 1:
            output = downblock(output, channels=channels * (2 ** i), \
                            update_collection=update_collection, name='downblock%d' % i)
            i += 1
        return output

def conditional_output(inputs, encode, update_collection, name='cond_out'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output1 = tf.layers.flatten(inputs)
        output2 = tf.concat([output1, encode], axis=1)
        output1 = dense(output1, units=1, update_collection=update_collection, name='output1')
        output2 = dense(output2, units=1, update_collection=update_collection, name='output2')
        return (output1 + output2) / 2
    
def discriminator1(inputs, encode, update_collection, name='Discriminator1'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs, update_collection=update_collection)
        return conditional_output(inputs=down, encode=encode, update_collection=update_collection)
    
def discriminator2(inputs, encode, update_collection, name='Discriminator2'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs, update_collection=update_collection)
        return conditional_output(inputs=down, encode=encode, update_collection=update_collection)
    
def discriminator3(inputs, encode, update_collection, name='Discriminator3'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        down = downsample(inputs=inputs, update_collection=update_collection)
        return conditional_output(inputs=down, encode=encode, update_collection=update_collection)

def get_noise(size):
    return normal(loc=0.0, scale=1.0, size=size)
