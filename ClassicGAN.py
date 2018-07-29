from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import random
import warnings
import tensorflow as tf
from tensorflow.contrib import data
from tensorflow.python.client import timeline # pylint: disable=E0611
import numpy as np
import argparse
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH
from Model import (get_noise, generator1, generator2, generator3, 
                    discriminator1, discriminator2, discriminator3, 
                    process, shared_gen, encoder, NOISE_LENGTH, NO_OPS, 
                    SPECTRAL_UPDATE_OPS)
from Convert import unpack_sample
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TOTAL_TRAIN_EPOCH = 100
LAMBDA = 10
TRAIN_RATIO_DIS = 1
TRAIN_RATIO_GEN = 1
EPSILON = 1e-8
DRIFT = 1e-3
BATCH_SIZE = 2

def gradient_penalty(real, gen, encode, discriminator):
    with tf.name_scope('gradient_penalty'):
        alpha = tf.random_uniform(shape=[BATCH_SIZE] + [1] * (gen.shape.ndims - 1), minval=0., maxval=1.)
        interpolate = real + alpha * (gen - real)
        gradients = tf.gradients(
            discriminator(inputs=interpolate, encode=encode, update_collection=NO_OPS), interpolate
        )[0]
        slopes = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(gradients), axis=list(range(1, gradients.shape.ndims))))
        output = tf.reduce_mean((slopes - 1.) ** 2)
        return LAMBDA * output

def lipschitz_penalty(real, gen, encode, discriminator):
    with tf.name_scope('lipschitz_penalty'):
        alpha = tf.random_uniform(shape=[BATCH_SIZE] + [1] * (gen.shape.ndims - 1), minval=0., maxval=1.)
        interpolate = real + alpha * (gen - real)
        gradients = tf.gradients(
            discriminator(inputs=interpolate, encode=encode, update_collection=NO_OPS), interpolate
        )[0]
        slopes = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(gradients), axis=list(range(1, gradients.shape.ndims))))
        output = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        return LAMBDA * output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--sample', type=str, default='', help='Samples based on input song. Empty string means training.'
    )
    parser.add_argument(
        '-c', '--concat', dest='concat', action='store_true', help='Enable concatenation.'
    )
    parser.add_argument(
        '--no-concat', dest='concat', action='store_false', help='Disable concatenation.'
    )
    parser.set_defaults(concat=False)
    parser.add_argument(
        '-r', '--record', dest='record', action='store_true', help='Enable recording.'
    )
    parser.add_argument(
        '--no-record', dest='record', action='store_false', help='Disable recording.'
    )
    parser.set_defaults(record=False) # Warning: Windows kills python if enabled.
    args = parser.parse_args()
    sampling = args.sample != ''

    if not os.path.exists('Checkpoints_v1'):
        os.makedirs('Checkpoints_v1')
    if not os.path.exists('Logs_v1'):
        os.makedirs('Logs_v1')
    if not os.path.exists('Samples_v1'):
        os.makedirs('Samples_v1')
    if not os.path.exists('Timelines_v1'):
        os.makedirs('Timelines_v1')
    
    filename = 'Dataset/cond_dataset.tfrecord'
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
    def _parse(example_proto):
        feature = {
            'label': tf.FixedLenFeature([], tf.int64), 
            'data': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(example_proto, feature)
        data = tf.decode_raw(parsed['data'], tf.uint8)
        label = tf.cast(parsed['label'], tf.uint8)
        data = tf.py_func(func=np.unpackbits, inp=[data], Tout=tf.uint8)
        label = tf.py_func(func=np.unpackbits, inp=[label], Tout=tf.uint8)
        data = tf.cast(data, tf.float32)
        label = tf.cast(label, tf.float32)
        data = tf.reshape(data, [CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH])
        label.set_shape([8])
        label = label[:CHANNEL_NUM]
        data = data * 2 - 1
        return {'data': data, 'label': label}

    dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=16384))
    dataset = dataset.apply(
        data.map_and_batch(_parse, batch_size=BATCH_SIZE, num_parallel_batches=16, drop_remainder=True)
    )
    dataset = dataset.prefetch(32)
    dataset = dataset.apply(data.prefetch_to_device('/gpu:0'))
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    real_input_3 = next_data['data']
    label = next_data['label']

    input_noise = tf.placeholder(dtype=tf.float32, shape=[None, NOISE_LENGTH], name='input_noise')

    real_input_2 = tf.layers.average_pooling2d(
        inputs=real_input_3, 
        pool_size=2, 
        strides=2, 
        padding='same', 
        data_format='channels_first', 
        name='real_input_2'
    )
    real_input_2 -= tf.reduce_min(real_input_2)
    real_input_2 /= (tf.reduce_max(real_input_2) + EPSILON)
    real_input_2 = 2 * real_input_2 - 1
    real_input_1 = tf.layers.average_pooling2d(
        inputs=real_input_2, 
        pool_size=2, 
        strides=2, 
        padding='same', 
        data_format='channels_first', 
        name='real_input_1'
    )
    real_input_1 -= tf.reduce_min(real_input_1)
    real_input_1 /= (tf.reduce_max(real_input_1) + EPSILON)
    real_input_1 = 2 * real_input_1 - 1
    train = tf.placeholder(dtype=tf.bool, name='train')
    # shape: [None, 6, 64, 256]
    encode = encoder(inputs=real_input_3, update_collection=SPECTRAL_UPDATE_OPS, train=train)
    # shape: [None, 64]
    tf.summary.histogram('encode', encode)
    print('Encoder set')

    real_input_3_image = tf.expand_dims(real_input_3[:1], axis=-1, name='real_input_3_image')
    # shape: [1, 6, 128, 512, 1]
    real_input_2_image = tf.expand_dims(real_input_2[:1], axis=-1, name='real_input_2_image')
    # shape: [1, 6, 64, 256, 1]
    real_input_1_image = tf.expand_dims(real_input_1[:1], axis=-1, name='real_input_1_image')
    # shape: [1, 6, 32, 128, 1]
    for i in range(CHANNEL_NUM):
        tf.summary.image('real_input_1_%d' % i, real_input_1_image[:, i])
        tf.summary.image('real_input_2_%d' % i, real_input_2_image[:, i])
        tf.summary.image('real_input_3_%d' % i, real_input_3_image[:, i])

    shared_output = shared_gen(
        noise=input_noise, 
        label=label, 
        update_collection=SPECTRAL_UPDATE_OPS, 
        train=train
    )
    # shape: [None, 64, 16, 64]
    gen1 = generator1(
        inputs=shared_output, 
        label=label, 
        update_collection=SPECTRAL_UPDATE_OPS, 
        train=train
    )
    output_gen1 = process(gen1, 1, train, SPECTRAL_UPDATE_OPS) * tf.expand_dims(tf.expand_dims(label, axis=-1), axis=-1)
    # shape: [None, 6, 32, 128]
    gen2 = generator2(
        inputs=gen1, 
        label=label, 
        update_collection=SPECTRAL_UPDATE_OPS, 
        train=train
    )
    output_gen2 = process(gen2, 2, train, SPECTRAL_UPDATE_OPS) * tf.expand_dims(tf.expand_dims(label, axis=-1), axis=-1)
    # shape: [None, 6, 64, 256]
    gen3 = generator3(
        inputs=gen2, 
        label=label, 
        update_collection=SPECTRAL_UPDATE_OPS, 
        train=train
    )
    output_gen3 = process(gen3, 3, train, SPECTRAL_UPDATE_OPS) * tf.expand_dims(tf.expand_dims(label, axis=-1), axis=-1)
    # shape: [None, 6, 128, 512]
    print('Generators set')
    dis1_real = discriminator1(inputs=real_input_1, encode=encode, update_collection=SPECTRAL_UPDATE_OPS)
    dis1_gen = discriminator1(inputs=output_gen1, encode=encode, update_collection=NO_OPS)
    dis2_real = discriminator2(inputs=real_input_2, encode=encode, update_collection=SPECTRAL_UPDATE_OPS)
    dis2_gen = discriminator2(inputs=output_gen2, encode=encode, update_collection=NO_OPS)
    dis3_real = discriminator3(inputs=real_input_3, encode=encode, update_collection=SPECTRAL_UPDATE_OPS)
    dis3_gen = discriminator3(inputs=output_gen3, encode=encode, update_collection=NO_OPS)
    print('Discriminators set')
    loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real) + lipschitz_penalty(
        real=real_input_1, gen=output_gen1, encode=encode, discriminator=discriminator1
    ) + DRIFT * tf.reduce_mean(tf.square(dis1_real))
    loss_gen1 = -tf.reduce_mean(dis1_gen)
    loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real) + lipschitz_penalty(
        real=real_input_2, gen=output_gen2, encode=encode, discriminator=discriminator2
    ) + DRIFT * tf.reduce_mean(tf.square(dis2_real))
    loss_gen2 = -tf.reduce_mean(dis2_gen)
    loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real) + lipschitz_penalty(
        real=real_input_3, gen=output_gen3, encode=encode, discriminator=discriminator3
    ) + DRIFT * tf.reduce_mean(tf.square(dis3_real))
    loss_gen3 = -tf.reduce_mean(dis3_gen)
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
    tf.summary.scalar('loss_gen', loss_gen)
    print('Losses set')
    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Shared_generator')
    #gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    for i in range(CHANNEL_NUM):
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator1_%d' % i)
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator2_%d' % i)
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator3_%d' % i)
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')
    dis1_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2')
    dis2_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3')
    dis3_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Shared_generator')
    #gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    for i in range(CHANNEL_NUM):
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator1_%d' % i)
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator2_%d' % i)
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator3_%d' % i)
    dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1')
    dis1_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2')
    dis2_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3')
    dis3_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    spectral_norm_update_ops = tf.get_collection(SPECTRAL_UPDATE_OPS)
    with tf.name_scope('optimizers'):
        with tf.control_dependencies(dis1_extra_update_ops):
            dis1_train = tf.train.AdamOptimizer(
                learning_rate=0.0004, beta1=0.5, beta2=0.9
            ).minimize(
                loss=loss_dis1, var_list=dis1_var, name='dis1_train'
            )
        print('dis1_train setup')
        with tf.control_dependencies(dis2_extra_update_ops):
            dis2_train = tf.train.AdamOptimizer(
                learning_rate=0.0004, beta1=0.5, beta2=0.9
            ).minimize(
                loss=loss_dis2, var_list=dis2_var, name='dis2_train'
            )
        print('dis2_train setup')
        with tf.control_dependencies(dis3_extra_update_ops):
            dis3_train = tf.train.AdamOptimizer(
                learning_rate=0.0004, beta1=0.5, beta2=0.9
            ).minimize(
                loss=loss_dis3, var_list=dis3_var, name='dis3_train'
            )
        print('dis3_train setup')
        with tf.control_dependencies(gen_extra_update_ops):
            gen_train = tf.train.AdamOptimizer(
                learning_rate=0.0001, beta1=0.5, beta2=0.9
            ).minimize(
                loss=loss_gen, var_list=gen_var, name='gen_train'
            )
        print('gen_train setup')
    print('Optimizers set')
    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        if tf.train.latest_checkpoint('Checkpoints_v1') is not None:
            print('Restoring...')
            saver.restore(sess, tf.train.latest_checkpoint('Checkpoints_v1'))
        feed_dict = {input_noise: None, train: True}
        print('preparing complete')
        if sampling:
            feed_dict = {input_noise: None, real_input_3: None, train: False}
            path = args.sample
            try:
                feed_dict[real_input_3] = roll(path)[:BATCH_SIZE]
            except:
                print('Error while opening file.')
                exit()
            feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
            samples = sess.run(output_gen3, feed_dict=feed_dict)
            path = path.split('/')[-1]
            if not os.path.exists('Samples_v1/sample_%s'):
                os.mkdir('Samples_v1/sample_%s' % path)
            np.save(file='Samples_v1/sample_%s' % path + '/%s' % path, arr=samples)
            unpack_sample(name='Samples_v1/sample_%s' % path + '/%s.npy' % path, concat=args.concat)
            exit()
        writer = tf.summary.FileWriter('Logs_v1', sess.graph)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        epoch_num = 100001
        for train_count in tqdm(range(epoch_num)):
            for i in range(TRAIN_RATIO_DIS):
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                feed_dict[train] = True
                *_, loss_val_dis1, loss_val_dis2, loss_val_dis3 = sess.run([
                        dis1_train, dis2_train, dis3_train, loss_dis1, loss_dis2, loss_dis3
                    ], feed_dict=feed_dict, options=run_options
                )
            for i in range(TRAIN_RATIO_GEN):
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                feed_dict[train] = True
                summary, _, loss_val_gen = sess.run([
                        merged, gen_train, loss_gen
                    ], feed_dict=feed_dict, options=run_options
                )
            sess.run(spectral_norm_update_ops)
            writer.add_summary(summary, train_count)
            tqdm.write('%06d' % train_count, end=' ')
            tqdm.write('Discriminator1 loss : %.7f' % loss_val_dis1, end=' ')
            tqdm.write('Discriminator2 loss : %.7f' % loss_val_dis2, end=' ')
            tqdm.write('Discriminator3 loss : %.7f' % loss_val_dis3, end=' ')
            tqdm.write('Generator loss : %.7f' % loss_val_gen)
            if train_count % 1000 == 0:
                feed_dict[input_noise] = get_noise([BATCH_SIZE, NOISE_LENGTH])
                feed_dict[train] = False
                samples = sess.run(output_gen3, feed_dict=feed_dict)
                np.save(file='Samples_v1/song_%06d' % train_count, arr=samples)
                unpack_sample('Samples_v1/song_%06d' % train_count)
                save_path = saver.save(sess, 'Checkpoints_v1/song_%06d' % train_count + '.ckpt')
                tqdm.write('Model Saved: %s' % save_path)
                if args.record:
                    trace_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # pylint: disable=E1101
                    run_metadata = tf.RunMetadata()
                    sess.run([
                            dis1_train, dis2_train, dis3_train, gen_train
                        ], feed_dict=feed_dict, options=trace_options, run_metadata=run_metadata
                    )
                    writer.add_run_metadata(run_metadata, 'run_%d' % train_count)
                    tl = timeline.Timeline(run_metadata.step_stats) # pylint: disable=E1101
                    ctf = tl.generate_chrome_trace_format()
                    with open('Timelines_v1/timeline_%d.json' % train_count, 'w') as f:
                        f.write(ctf)
        writer.close()

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
