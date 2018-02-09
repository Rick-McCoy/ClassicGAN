from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import memory_saving_gradients
from Model import conv2d
# monkey patch memory_saving_gradients.gradients_speed to point to our custom version, with automatic
# checkpoint selection
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
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

def encoder(inputs, num, train):
    with tf.variable_scope('Encoder'+str(num)):
        # shape: [BATCH_NUM, CLASS_NUM, INPUT_LENGTH]
        inputs = tf.expand_dims(inputs, axis=1)
        # shape: [BATCH_NUM, 1, CLASS_NUM, INPUT_LENGTH)
        conv1 = conv2d(inputs=inputs, filters=16, training=train, kernel_size=[2, 1], strides=(2, 1), name='conv1')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 2, INPUT_LENGTH]
        conv2 = conv2d(inputs=conv1, filters=16, training=train, kernel_size=[3, 1], strides=(3, 1), name='conv2')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 6, INPUT_LENGTH]
        conv3 = conv2d(inputs=conv2, filters=16, training=train, kernel_size=[3, 1], strides=(3, 1), name='conv3')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH]
        conv4 = conv2d(inputs=conv3, filters=16, training=train, kernel_size=[1, 2], strides=(1, 2), name='conv4')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH // 3]
        conv5 = conv2d(inputs=conv4, filters=16, training=train, kernel_size=[1, 3], strides=(1, 3), name='conv5')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH // 12]
        conv6 = conv2d(inputs=conv5, filters=16, training=train, kernel_size=[1, 4], strides=(1, 4), name='conv6')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH // 48]
        conv7 = conv2d(inputs=conv6, filters=16, training=train, kernel_size=[1, 4], strides=(1, 4), name='conv7')
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH // 192]
        return conv7

def decoder(inputs, num, train):
    with tf.variable_scope('Decoder'+str(num)):
        # shape: [BATCH_NUM, 16, CLASS_NUM // 18, INPUT_LENGTH // 192]
        deconv1 = conv2d_transpose(inputs=inputs, filters=16, strides=(1, 4), training=train, name='deconv1')
        # shape: [BATHC_NUM, 16, CLASS_NUM // 18, INPUT_LENGHT // 48]
        deconv2 = conv2d_transpose(inputs=deconv1, filters=16, strides=(1, 4), training=train, name='cdeconv2')
