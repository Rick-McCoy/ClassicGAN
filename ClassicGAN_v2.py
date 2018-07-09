from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import datetime
import tensorflow as tf
from tensorflow.contrib import data
from tensorflow.python.client import timeline # pylint: disable=E0611
import numpy as np
import argparse
from tqdm import tqdm, trange
from ops import (conv2d, conv2d_transpose, pool, dense_layer, 
                leaky_relu, minibatch_stddev, pixelwise_norm, resize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ClassicGAN:
    def __init__(self, 
            learning_rate_d=0.0004, 
            learning_rate_g=0.0001, 
            p_lambda=10.0, 
            p_gamma=1.0, 
            epsilon=0.001, 
            z_length=512, 
            n_imgs=160000, 
            lipschitz_penalty=True, 
            args=None
        ):
        self.channels = [512, 512, 512, 512, 256, 128, 64, 32]
        self.batch_size = [64, 128, 64, 32, 16, 16, 8, 4]
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.p_lambda = p_lambda
        self.p_gamma = p_gamma
        self.epsilon = epsilon
        self.z_length = z_length
        self.n_imgs = n_imgs
        self.lipschitz_penalty = lipschitz_penalty
        self.z = tf.placeholder(tf.float32, [None, self.z_length])
        self.channel_num = 6
        self.class_num = 128
        self.input_length = 512
        self.sampling = args.sample != ''
        self.record = args.record

        with tf.variable_scope('image_count'):
            self.total_imgs = tf.Variable(0.0, name='image_step', trainable=False)
            self.img_count_placeholder = tf.placeholder(tf.float32)
            self.img_step_op = tf.assign(self.total_imgs, 
                tf.add(self.total_imgs, self.img_count_placeholder))
            self.img_step = tf.mod(tf.add(self.total_imgs, self.n_imgs), self.n_imgs * 2)
            self.alpha = tf.minimum(1.0, tf.div(self.img_step, self.n_imgs))
            self.layer = tf.floor_div(tf.add(self.total_imgs, self.n_imgs), self.n_imgs * 2)

        self.get_dataset = self.make_dataset()
        self.x = self.next_batch()

        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_g, beta1=0, beta2=0.9)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_d, beta1=0, beta2=0.9)

        self.n_layers = 7
        self.global_step = tf.train.get_or_create_global_step()
        self.networks = [self._create_network(i + 1) for i in range(self.n_layers)]
        print('Networks set.')
        self.GPU_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        self.config = tf.ConfigProto(allow_soft_placement=True, gpu_options=self.GPU_options)
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(logdir='Logs', graph=self.sess.graph)
        self.saver = tf.train.Saver()

        if tf.train.latest_checkpoint('Checkpoints') is not None:
            print('Restoring...')
            self.saver.restore(self.sess, tf.train.latest_checkpoint('Checkpoints'))
            print('Completely restored.')

        print('Initialization complete.')


    def _reparameterize(self, x0, x1):
        return tf.add(
            tf.scalar_mul(tf.subtract(1.0, self.alpha), x0), 
            tf.scalar_mul(self.alpha, x1)
        )

    def _create_network(self, layers):

        def generator(z):
            z = z[:self.batch_size[layers]]
            with tf.variable_scope('Generator'):
                with tf.variable_scope('latent_vector'):
                    g1 = tf.expand_dims(tf.expand_dims(z, axis=-1), axis=-1)
                for i in range(layers):
                    with tf.variable_scope('layer_{}'.format(i)):
                        if i > 0:
                            g1 = resize(g1)
                        if i == layers - 1 and layers > 1:
                            g0 = g1
                        with tf.variable_scope('1'):
                            if i == 0:
                                g1 = pixelwise_norm(leaky_relu(conv2d_transpose(
                                    g1, [tf.shape(g1)[0], self.channels[0], 2, 8]
                                )))
                            else:
                                g1 = pixelwise_norm(leaky_relu(conv2d(g1, self.channels[i])))
                        with tf.variable_scope('2'):
                            g1 = pixelwise_norm(leaky_relu(conv2d(g1, self.channels[i])))
                with tf.variable_scope('rgb_layer_{}'.format(layers - 1)):
                    g1 = conv2d(g1, 6, 1, weight_norm=False)
                if layers > 1:
                    with tf.variable_scope('rgb_layer_{}'.format(layers - 2)):
                        g0 = conv2d(g0, 6, 1, weight_norm=False)
                        g = self._reparameterize(g0, g1)
                else:
                    g = g1
            return tf.tanh(g)

        def discriminator(x):
            with tf.variable_scope('Discriminator'):
                if layers > 1:
                    with tf.variable_scope('rgb_layer_{}'.format(layers - 2)):
                        d0 = pool(x)
                        d0 = leaky_relu(conv2d(d0, self.channels[layers - 1], 1))
                with tf.variable_scope('rgb_layer_{}'.format(layers - 1)):
                    d1 = leaky_relu(conv2d(x, self.channels[layers], 1))
                for i in reversed(range(layers)):
                    with tf.variable_scope('layer_{}'.format(i)):
                        if i == 0:
                            d1 = minibatch_stddev(d1)
                        with tf.variable_scope('1'):
                            d1 = leaky_relu(conv2d(d1, self.channels[i]))
                        with tf.variable_scope('2'):
                            if i == 0:
                                d1 = leaky_relu(conv2d(d1, self.channels[0], 2, 2))
                            else:
                                d1 = leaky_relu(conv2d(d1, self.channels[i]))
                        if i > 0:
                            d1 = pool(d1)
                        if i == layers - 1 and layers > 1:
                            d1 = self._reparameterize(d0, d1)
                with tf.variable_scope('dense'):
                    d = dense_layer(tf.layers.flatten(d1), 1)
            return d

        dim1, dim2 = 2 ** layers, 2 ** (layers + 2)

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            G = generator(self.z)
            Dz = discriminator(G)
            with tf.variable_scope('reshape'):
                if layers > 1:
                    x0 = resize(self.x, (dim1 // 2, dim2 // 2))
                    x0 = resize(x0, (dim1, dim2))
                    x1 = resize(self.x, (dim1, dim2))
                    x = self._reparameterize(x0, x1)
                else:
                    x = resize(self.x, (dim1, dim2))
            Dx = discriminator(x[:Dz.get_shape().as_list()[0]])

            alpha = tf.random_uniform(shape=[tf.shape(Dz)[0], 1, 1, 1], minval=0., maxval=1.)
            interpolate = alpha * x + (1 - alpha) * G
            D_inter = discriminator(interpolate)

        with tf.variable_scope('Loss_function'):

            WD = Dz - Dx

            gradients = tf.gradients(D_inter, [interpolate])[0]
            slopes = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(gradients), axis=list(range(1, gradients.shape.ndims))))
            if self.lipschitz_penalty:
                GP = tf.square(tf.maximum((slopes - self.p_gamma) / self.p_gamma, 0))
            else:
                GP = tf.square((slopes - self.p_gamma) / self.p_gamma)
            GP_scaled = self.p_lambda * GP

            drift = self.epsilon * tf.square(Dx)

            g_loss = tf.reduce_mean(-Dz)
            d_loss = tf.reduce_mean(WD + GP_scaled + drift)
            WD = tf.reduce_mean(WD)
            GP = tf.reduce_mean(GP)
            
            WD_sum = tf.summary.scalar('Wasserstein_distance_{}x{}'.format(dim1, dim2), WD)
            GP_sum = tf.summary.scalar('gradient_penalty_{}x{}'.format(dim1, dim2), GP)

        g_vars, d_vars = [], []
        var_scopes = ['layer_{}'.format(i) for i in range(layers)]
        var_scopes += ['dense', 'rgb_layer_{}'.format(layers - 1), 'rgb_layer_{}'.format(layers - 2)]
        for scope in var_scopes:
            g_vars += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope='Network/Generator/{}'.format(scope)
            )
            d_vars += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope='Network/Discriminator/{}'.format(scope)
            )

        with tf.variable_scope('Optimize'):
            g_train = self.g_optimizer.minimize(g_loss, var_list=g_vars, global_step=self.global_step)
            d_train = self.d_optimizer.minimize(d_loss, var_list=d_vars)
        
        with tf.variable_scope('Sample'):
            fake_imgs = G[:1]
            real_imgs = x[:1]

            if dim2 < self.input_length:
                fake_imgs = resize(fake_imgs, (self.class_num, self.input_length))
                real_imgs = resize(real_imgs, (self.class_num, self.input_length))

            fake_img_sum = [tf.summary.image(
                'fake{}x{}'.format(dim1, dim2), tf.expand_dims(fake_imgs[:, i], axis=-1)
                ) for i in range(self.channel_num)]
            real_img_sum = [tf.summary.image(
                'real{}x{}'.format(dim1, dim2), tf.expand_dims(real_imgs[:, i], axis=-1)
                ) for i in range(self.channel_num)]

        print('Stage {}x{} setup complete.'.format(dim1, dim2))

        return (dim1, dim2, WD, GP, WD_sum, GP_sum, g_train, d_train, 
                fake_img_sum, real_img_sum, G, discriminator)

    def _add_summary(self, string, global_step):
        self.writer.add_summary(string, global_step)

    def _z(self, batch_size):
        return np.random.normal(loc=0.0, scale=1.0, size=[batch_size, self.z_length])

    def make_dataset(self):
        self.filename = 'Dataset/dataset.tfrecord'
        self.dataset = tf.data.TFRecordDataset(self.filename, num_parallel_reads=8)
        def _parse(example_proto):
            feature = {'roll' : tf.FixedLenFeature([], tf.string)}
            parsed = tf.parse_single_example(example_proto, feature)
            data = tf.decode_raw(parsed['roll'], tf.uint8)
            data = tf.py_func(func=np.unpackbits, inp=[data], Tout=tf.uint8)
            data = tf.cast(data, tf.float32)
            data = tf.reshape(data, [self.channel_num, self.class_num, self.input_length])
            data = data * 2 - 1
            return data
        self.dataset = self.dataset.apply(data.shuffle_and_repeat(buffer_size=16384))
        self.dataset = self.dataset.apply(data.map_and_batch(_parse, batch_size=128, 
                                            num_parallel_batches=16, drop_remainder=True))
        self.dataset = self.dataset.prefetch(128)
        self.iterator = self.dataset.make_one_shot_iterator()
        batch = self.iterator.get_next()

        return batch

    def next_batch(self):
        batch = self.get_dataset
        return batch

    def train(self):

        total_imgs = self.sess.run(self.total_imgs)
        prev_layer = 0
        running_average_time = None

        while total_imgs < (self.n_layers - 0.5) * self.n_imgs * 2:
            start_time = datetime.datetime.now()

            layer, gs, alpha, total_imgs = self.sess.run([
                self.layer, self.global_step, self.alpha, self.total_imgs
            ])
            layer = int(layer)
            if layer != prev_layer:
                running_average_time = None
                prev_layer = layer

            save_interval = max(1000, 10000 // 2 ** layer)

            (dim1, dim2, WD, GP, WD_sum, GP_sum, g_train, d_train, 
            fake_img_sum, real_img_sum, *_) = self.networks[layer]
            feed_dict = {self.z: self._z(self.batch_size[layer])}

            self.sess.run(g_train, feed_dict)
            self.sess.run(d_train, feed_dict)

            WD_, GP_, WD_sum_str, GP_sum_str, fake_img_sum_str, \
            real_img_sum_str = self.sess.run([
                WD, GP, WD_sum, GP_sum, fake_img_sum, real_img_sum
            ], feed_dict)

            current_time = datetime.datetime.now()
            if running_average_time is not None:
                running_average_time = running_average_time * 0.8 + (current_time - start_time) * 0.2
            else:
                running_average_time = current_time - start_time
            total_step = self.n_imgs * 2 // self.batch_size[layer]
            percentage = ((gs + 1) % total_step) / total_step * 100
            remain_time = (1 - percentage / 100) * running_average_time * total_step

            print('Step: {}, size: {}x{}, alpha: {:.7f}, WD: {:.7f}, GP: {:.7f}, percentage: {:.2f}%, remaining time: {}'.format(
                gs, dim1, dim2, alpha, WD_, GP_, percentage, remain_time
            ))

            if gs % 20 == 0:
                self._add_summary(WD_sum_str, gs)
                self._add_summary(GP_sum_str, gs)
                for i in range(self.channel_num):
                    self._add_summary(fake_img_sum_str[i], gs)
                    self._add_summary(real_img_sum_str[i], gs)

            if gs % save_interval == 0:
                print('Saving...')
                self.saver.save(self.sess, 'Checkpoints/{}.ckpt'.format(gs))
                print('Model saved.')
                if self.record:
                    trace_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # pylint: disable=E1101
                    run_metadata = tf.RunMetadata()
                    self.sess.run([g_train, d_train], feed_dict=feed_dict, options=trace_options, run_metadata=run_metadata)
                    self.writer.add_run_metadata(run_metadata, 'run_%d' % gs)
                    tl = timeline.Timeline(run_metadata.step_stats) # pylint: disable=E1101
                    ctf = tl.generate_chrome_trace_format()
                    with open('Timelines/%d.json' % gs, 'w') as f:
                        f.write(ctf)

            img_count = self.batch_size[layer]
            self.sess.run(self.img_step_op, {self.img_count_placeholder: img_count})

        self.sess.close()
        print('Training complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', type=str, default='', \
                        help='Samples based on input song. Empty string means training.')
    parser.add_argument('-r', '--record', dest='record', action='store_true', help='Enable recording.')
    parser.add_argument('--no-record', dest='record', action='store_false', help='Disable recording.')
    parser.set_defaults(record=False) # Warning: Windows kills python if enabled.
    args = parser.parse_args()

    if not os.path.exists('Logs'):
        os.mkdir('Logs')
    if not os.path.exists('Checkpoints'):
        os.mkdir('Checkpoints')
    if not os.path.exists('Timelines'):
        os.mkdir('Timelines')
    if not os.path.exists('Samples'):
        os.mkdir('Samples')

    classicgan = ClassicGAN(args=args)
    classicgan.train()

if __name__ == '__main__':
    main()

