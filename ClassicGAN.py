from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import random
import warnings
import tensorflow as tf
from tensorflow.contrib import data
import numpy as np
import argparse
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_SIZE
from Model import get_noise, generator1, generator2, generator3, \
                    discriminator1, discriminator2, discriminator3, \
                    shared_gen, encoder, NOISE_LENGTH
from Convert import unpack_sample
import memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TOTAL_TRAIN_EPOCH = 100
LAMBDA = 10
#LAMBDA1 = 1
#LAMBDA2 = 5
TRAIN_RATIO_DIS = 5
TRAIN_RATIO_GEN = 1

def gradient_penalty(real, gen, encode, discriminator):
    with tf.name_scope('gradient_penalty'):
        alpha = tf.random_uniform(shape=[BATCH_SIZE] + [1] * (gen.shape.ndims - 1), minval=0., maxval=1.)
        interpolate = real + alpha * (gen - real)
        gradients = tf.gradients(discriminator(inputs=interpolate, encode=encode), interpolate)[0]
        slopes = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(gradients), axis=list(range(1, gradients.shape.ndims))))
        output = tf.reduce_mean((slopes - 1.) ** 2)
        return LAMBDA * output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', type=str, default='', \
                                            help='Samples based on input song. Empty string means training.')
    parser.add_argument('-c', '--concat', type=bool, default=False, \
                                            help='Enable Concatenation. Defaults to False.')
    args = parser.parse_args()
    sampling = args.sample != ''

    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('Samples'):
        os.makedirs('Samples')
    
    with tf.name_scope('inputs'):
        filename = 'Dataset/dataset.tfrecord'
        dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
        def _parse(example_proto):
            feature = {'roll' : tf.FixedLenFeature((6, 72, 384), tf.float32)}
            parsed = tf.parse_single_example(example_proto, feature)
            return parsed['roll']
        dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=16384))
        dataset = dataset.apply(data.map_and_batch(_parse, batch_size=BATCH_SIZE\
                                        , num_parallel_batches=8, drop_remainder=True))
        dataset = dataset.prefetch(4)
        iterator = dataset.make_one_shot_iterator()
        real_input_3 = iterator.get_next()

        input_noise = tf.placeholder(dtype=tf.float32, shape=[None, NOISE_LENGTH], name='input_noise')

        train = tf.placeholder(dtype=tf.bool, name='traintest')
        #real_input_4 = tf.placeholder(dtype=tf.float32, \
        # shape=[None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH], name='real_input_4')
        real_input_3_split = tf.split(real_input_3, num_or_size_splits=4, axis=-1, name='real_input_3_split')
        # shape: [4, None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_2 = tf.stack(real_input_3_split, axis=2, name='real_input_2')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_1 = tf.layers.max_pooling3d(inputs=real_input_2, pool_size=[1, 2, 2], strides=(1, 2, 2), \
                                                    padding='same', data_format='channels_first', name='real_input_1')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        encode = encoder(inputs=real_input_2, train=train)
        # shape: [None, 4, 2, 8]

        real_input_3_image = tf.expand_dims(real_input_3[:1], axis=-1, name='real_input_3_image_expand')
        # shape: [1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
        real_input_3_image = tf.unstack(real_input_3_image, axis=1, name='real_input_3_image_unstack')
        # shape: [CHANNEL_NUM, 1, CLASS_NUM, INPUT_LENGTH, 1]
        for i, j in enumerate(real_input_3_image):
            tf.summary.image('real_input_3_' + str(i), j)
            
        real_input_2_image = [tf.layers.max_pooling2d(inputs=image, pool_size=[2, 2], strides=(2, 2), \
                                                        padding='same', name='real_input_2_inage') \
                                                        for image in real_input_3_image]
        # shape: [CHANNEL_NUM, 1, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        for i, j in enumerate(real_input_2_image):
            tf.summary.image('real_input_2_' + str(i), j)
            
        real_input_1_image = [tf.layers.max_pooling2d(inputs=image, pool_size=[2, 2], strides=(2, 2), \
                                                        padding='same', name='real_input_1_inage') \
                                                        for image in real_input_2_image]
        # shape: [CHANNEL_NUM, 1, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        for i, j in enumerate(real_input_1_image):
            tf.summary.image('real_input_1_' + str(i), j)

    with tf.name_scope('generator'):
        shared_output = shared_gen(noise=input_noise, encode=encode, train=train)
        # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        output_gen1, gen1 = zip(*[generator1(inputs=shared_output, encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        # shape: [CHANNEL_NUM, None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        output_gen1 = tf.stack(output_gen1, axis=1, name='output_gen1_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        gen1 = [tf.concat(values=[i, output_gen1], axis=1) for i in gen1]
        # shape: [CHANNEL_NUM, None, 38, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        output_gen2, gen2 = zip(*[generator2(inputs=gen1[i], encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM, INPUT_LENGTH // 4]
        # shape: [CHANNEL_NUM, None, 16, 4, CLASS_NUM, INPUT_LENGTH // 4]
        output_gen2 = tf.stack(output_gen2, axis=1, name='output_gen2_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        gen2 = [tf.concat(values=[i, output_gen2], axis=1) for i in gen2]
        # shape: [CHANNEL_NUM, None, 22, 4, CLASS_NUM, INPUT_LENGTH // 4]
        output_gen3 = [generator3(inputs=gen2[i], encode=encode, \
                                    num=i, train=train) for i in range(CHANNEL_NUM)]
        # shape: [CHANNEL_NUM, None, CLASS_NUM, INPUT_LENGTH]
        output_gen3 = tf.stack(output_gen3, axis=1, name='output_gen3_stack')
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Generators set')
    with tf.name_scope('discriminator'):
        dis1_real = discriminator1(inputs=real_input_1, encode=encode)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        dis1_gen = discriminator1(inputs=output_gen1, encode=encode)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        dis2_real = discriminator2(inputs=real_input_2, encode=encode)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        dis2_gen = discriminator2(inputs=output_gen2, encode=encode)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        dis3_real = discriminator3(inputs=real_input_3, encode=encode)
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        dis3_gen = discriminator3(inputs=output_gen3, encode=encode)
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Discriminators set')
    with tf.name_scope('loss'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real) + gradient_penalty(real=real_input_1, \
                                        gen=output_gen1, encode=encode, discriminator=discriminator1)
        #mean_gen1, dev_gen1 = tf.nn.moments(output_gen1, axes=list(range(2, output_gen1.shape.ndims)))
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real) + gradient_penalty(real=real_input_2, \
                                        gen=output_gen2, encode=encode, discriminator=discriminator2)
        #mean_gen2, dev_gen2 = tf.nn.moments(output_gen2, axes=list(range(2, output_gen2.shape.ndims)))
        loss_gen2 = -tf.reduce_mean(dis2_gen)# + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen1, mean_gen2)) \
                                            #    + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen1, dev_gen2))
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real) + gradient_penalty(real=real_input_3, \
                                        gen=output_gen3, encode=encode, discriminator=discriminator3)
        #mean_gen3, dev_gen3 = tf.nn.moments(output_gen3, axes=list(range(2, output_gen3.shape.ndims)))
        loss_gen3 = -tf.reduce_mean(dis3_gen)# + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen2, mean_gen3)) \
                                            #    + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen2, dev_gen3))
        loss_gen = tf.add_n([loss_gen1, loss_gen2, loss_gen3]) / 3
        tf.summary.scalar('loss_dis1', loss_dis1)
        tf.summary.scalar('loss_gen1', loss_gen1)
        tf.summary.scalar('dis1_real', tf.reduce_mean(dis1_real))
        tf.summary.scalar('loss_dis2', loss_dis2)
        tf.summary.scalar('loss_gen2', loss_gen2)
        tf.summary.scalar('dis2_real', tf.reduce_mean(dis2_real))
        tf.summary.scalar('loss_dis3', loss_dis3)
        tf.summary.scalar('loss_gen3', loss_gen3)
        tf.summary.scalar('dis3_real', tf.reduce_mean(dis3_real))
    print('Losses set')
    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Shared_generator')
    gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    for i in range(CHANNEL_NUM):
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator1_' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator2_' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator3_' + str(i))
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2')
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3')
    gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Shared_generator')
    gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    for i in range(CHANNEL_NUM):
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator1_' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator2_' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator3_' + str(i))
    dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1')
    dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2')
    dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3')
    with tf.name_scope('optimizers'):
        with tf.control_dependencies(dis1_extra_update_ops):
            dis1_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(\
                                                    loss=loss_dis1, var_list=dis1_var, name='dis1_train')
        print('dis1_train setup')
        with tf.control_dependencies(dis2_extra_update_ops):
            dis2_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(\
                                                    loss=loss_dis2, var_list=dis2_var, name='dis2_train')
        print('dis2_train setup')
        with tf.control_dependencies(dis3_extra_update_ops):
            dis3_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(\
                                                    loss=loss_dis3, var_list=dis3_var, name='dis3_train')
        print('dis3_train setup')
        with tf.control_dependencies(gen_extra_update_ops):
            gen_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(\
                                                    loss=loss_gen, var_list=gen_var, name='gen_train')
        print('gen_train setup')
    print('Optimizers set')
    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        if tf.train.latest_checkpoint('Checkpoints') is not None:
            print('Restoring...')
            saver.restore(sess, tf.train.latest_checkpoint('Checkpoints'))
        feed_dict = {input_noise: None, train: True}
        print('preparing complete')
        if sampling:
            feed_dict = {input_noise: None, real_input_3: None, train: True}
            path = args.sample
            try:
                feed_dict[real_input_3] = roll(path)[:BATCH_SIZE]
            except:
                print('Error while opening file.')
                return
            feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
            feed_dict[train] = True
            samples = sess.run(output_gen3, feed_dict=feed_dict)
            path = path.split('/')[-1]
            if not os.path.exists('Samples/sample_%s'):
                os.mkdir('Samples/sample_%s' % path)
            np.save(file='Samples/sample_%s' % path + '/%s' % path, arr=samples)
            unpack_sample(name='Samples/sample_%s' % path+ '/%s.npy' % path, concat=args.concat)
            return
        writer = tf.summary.FileWriter('train', sess.graph)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        epoch_num = 1000000
        for train_count in tqdm(range(epoch_num)):
            feed_dict[train] = True
            for i in range(TRAIN_RATIO_DIS):
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                _, loss_val_dis1 = sess.run([dis1_train, loss_dis1], feed_dict=feed_dict, options=run_options)
                _, loss_val_dis2 = sess.run([dis2_train, loss_dis2], feed_dict=feed_dict, options=run_options)
                _, loss_val_dis3 = sess.run([dis3_train, loss_dis3], feed_dict=feed_dict, options=run_options)
            for i in range(TRAIN_RATIO_GEN):
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                summary, _, loss_val_gen = sess.run([merged, gen_train, loss_gen], \
                                                    feed_dict=feed_dict, options=run_options)
            writer.add_summary(summary, train_count)
            tqdm.write('%06d' % train_count, end=' ')
            tqdm.write('Discriminator1 loss : %.7f' % loss_val_dis1, end=' ')
            tqdm.write('Discriminator2 loss : %.7f' % loss_val_dis2, end=' ')
            tqdm.write('Discriminator3 loss : %.7f' % loss_val_dis3, end=' ')
            tqdm.write('Generator loss : %.7f' % loss_val_gen)
            if train_count % 1000 == 0:
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                samples = sess.run(output_gen3, feed_dict=feed_dict)
                np.save(file='Samples/song_%06d' % train_count, arr=samples)
                unpack_sample('Samples/song_%06d' % train_count)
                save_path = saver.save(sess, 'Checkpoints/song_%06d' % train_count + '.ckpt')
                tqdm.write('Model Saved: %s' % save_path)
        writer.close()
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
