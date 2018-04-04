from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import random
import warnings
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_SIZE
from Model import get_noise, generator1, generator2, generator3, generator4, \
                    noise_generator, time_seq_noise_generator, discriminator1_conditional, \
                    discriminator2_conditional, discriminator3_conditional, \
                    discriminator4_conditional, encoder, NOISE_LENGTH
from Convert import unpack_sample
import memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TOTAL_TRAIN_EPOCH = 100
LAMBDA = 10
LAMBDA1 = 2
LAMBDA2 = 10
TRAIN_RATIO_DIS = 5
TRAIN_RATIO_GEN = 1
pathlist = list(pathlib.Path('Dataset').glob('**/*.npy'))

def gradient_penalty(real, gen, encode, discriminator):
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
        dataset = tf.data.TFRecordDataset(filename)
        def _parse(example_proto):
            feature = {'roll' : tf.FixedLenFeature((6, 72, 384), tf.float32)}
            parsed = tf.parse_single_example(example_proto, feature)
            return parsed['roll']
        dataset = dataset.map(_parse).repeat().shuffle(buffer_size=10000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        iterator = dataset.make_one_shot_iterator()
        real_input_4 = iterator.get_next()

        #data = tf.placeholder(dtype=tf.float32, shape=[None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH])
        #dataset = tf.data.Dataset().from_tensor_slices(data).map(lambda x: x, num_parallel_calls=8)
        #dataset = dataset.repeat().shuffle(buffer_size=2000).apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        #iterator = dataset.make_initializable_iterator()
        #real_input_4 = iterator.get_next()

        input_noise1 = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, NOISE_LENGTH], name='input_noise1')
        input_noise2 = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, NOISE_LENGTH], name='input_noise2')
        input_noise3 = tf.placeholder(dtype=tf.float32, shape=[None, CHANNEL_NUM, 1, NOISE_LENGTH], name='input_noise3')
        input_noise4 = tf.placeholder(dtype=tf.float32, shape=[None, CHANNEL_NUM, 1, NOISE_LENGTH], name='input_noise4')

        train = tf.placeholder(dtype=tf.bool, name='traintest')
        noise1 = tf.tile(input=input_noise1, multiples=[1, CHANNEL_NUM, 4, 1], name='noise1')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        noise2_gen = noise_generator(noise=input_noise2, train=train)
        # shape: [None, 1, 4, NOISE_LENGTH]
        noise2 = tf.tile(input=noise2_gen, multiples=[1, CHANNEL_NUM, 1, 1], name='noise2')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        noise3 = tf.tile(input=input_noise3, multiples=[1, 1, 4, 1], name='noise3')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        input_noise4_split = tf.split(input_noise4, num_or_size_splits=CHANNEL_NUM, axis=1)
        # shape: [CHANNEL_NUM, None, 1, NOISE_LENGTH]
        noise4_split = [time_seq_noise_generator(noise=j, num=i, train=train) for i, j in enumerate(input_noise4_split)]
        # shape: [CHANNEL_NUM, None, 4, NOISE_LENGTH]
        noise4 = tf.concat(values=noise4_split, axis=1, name='noise4')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        #real_input_4 = tf.placeholder(dtype=tf.float32, \
        # shape=[None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH], name='real_input_4')
        real_input_4_split = tf.split(real_input_4, num_or_size_splits=4, axis=-1, name='real_input_4_split')
        # shape: [4, None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_3 = tf.stack(real_input_4_split, axis=2, name='real_input_3')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_2 = tf.layers.average_pooling3d(inputs=real_input_3, pool_size=[1, 2, 2], strides=(1, 2, 2), \
                                                    padding='same', data_format='channels_first', name='real_input_2')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        real_input_1 = tf.layers.average_pooling3d(inputs=real_input_2, pool_size=[1, 2, 2], strides=(1, 2, 2), \
                                                    padding='same', data_format='channels_first', name='real_input_1')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        real_input_3_split = tf.split(real_input_3, num_or_size_splits=CHANNEL_NUM, axis=1, name='real_input_3_split')
        # shape: [CHANNEL_NUM, None, 1, 4, CLASS_NUM, INPUT_LENGTH // 4]
        encode_split_mean, encode_split_var = zip(*[encoder(inputs=j, num=i, \
                                                    train=train) for i, j in enumerate(real_input_3_split)])
        # shape: [CHANNEL_NUM, None, 1, 4, 16]
        encode_mean = tf.concat(values=encode_split_mean, axis=1, name='encode_mean')
        # shape: [None, CHANNEL_NUM, 4, 16]
        encode_var = tf.concat(values=encode_split_var, axis=1, name='encode_var')
        # shape: [None, CHANNEL_NUM, 4, 16]
        input_noise = tf.concat(values=[noise1, noise2, noise3, noise4], axis=3, name='input_noise')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH * 4]

        real_input_4_image = tf.expand_dims(real_input_4[:BATCH_SIZE // 10], axis=-1, name='real_input_4_image_expand')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
        real_input_4_image = tf.unstack(real_input_4_image, axis=1, name='real_input_4_image_unstack')
        # shape: [CHANNEL_NUM, BATCH_SIZE // 10, CLASS_NUM, INPUT_LENGTH, 1]
        for i, j in enumerate(real_input_4_image):
            tf.summary.image('real_input_4_' + str(i), j)
        
        real_input_3_image = tf.unstack(real_input_3[:BATCH_SIZE // 10], axis=2, name='real_input_3_image_unstack')
        # shape: [4, BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_3_image = tf.concat(real_input_3_image, axis=-1, name='real_input_3_image_concat')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        real_input_3_image = tf.expand_dims(real_input_3_image, axis=-1, name='real_input_3_expand')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
        real_input_3_image = tf.unstack(real_input_3_image, axis=1, name='real_input_3_image')
        # shape: [CHANNEL_NUM, BATCH_SIZE // 10, CLASS_NUM, INPUT_LENGTH, 1]
        for i, j in enumerate(real_input_3_image):
            tf.summary.image('real_input_3_' + str(i), j)
            
        real_input_2_image = tf.unstack(real_input_2[:BATCH_SIZE // 10], axis=2, name='real_input_2_unstack')
        # shape: [4, BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 8]
        real_input_2_image = tf.concat(real_input_2_image, axis=-1, name='real_input_2_concat')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        real_input_2_image = tf.expand_dims(real_input_2_image, axis=-1, name='real_input_2_expand')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        real_input_2_image = tf.unstack(real_input_2_image, axis=1, name='real_input_2_image')
        # shape: [CHANNEL_NUM, BATCH_SIZE // 10, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        for i, j in enumerate(real_input_2_image):
            tf.summary.image('real_input_2_' + str(i), j)
            
        real_input_1_image = tf.unstack(real_input_1[:BATCH_SIZE // 10], axis=2, name='real_input_1_unstack')
        # shape: [4, BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 16]
        real_input_1_image = tf.concat(real_input_1_image, axis=-1, name='real_input_3_concat')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        real_input_1_image = tf.expand_dims(real_input_1_image, axis=-1, name='real_input_1_expand')
        # shape: [BATCH_SIZE // 10, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        real_input_1_image = tf.unstack(real_input_1_image, axis=1, name='real_input_1_image')
        # shape: [CHANNEL_NUM, BATCH_SIZE // 10, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        for i, j in enumerate(real_input_1_image):
            tf.summary.image('real_input_1_' + str(i), j)

    print('Inputs set')
    with tf.name_scope('generator'):
        input_noise_split = tf.unstack(input_noise, axis=1, name='input_noise_split')
        # shape: [CHANNEL_NUM, None, 4, NOISE_LENGTH * 4]
        encode_unstack_mean = tf.unstack(encode_mean, axis=1, name='encode_unstack_mean')
        # shape: [CHANNEL_NUM, None, 4, 16]
        encode_unstack_var = tf.unstack(encode_var, axis=1, name='encode_unstack_var')
        # shape: [CHANNEL_NUM, None, 4, 16]
        encode_unstack_normal = [tf.distributions.Normal(loc=encode_unstack_mean[i], \
                                            scale=encode_unstack_var[i]) for i in range(CHANNEL_NUM)]
        # shape: [CHANNEL_NUM, None, 4, 16]
        input_gen1, gen1 = zip(*[generator1(noise=input_noise_split[i], encode=encode_unstack_normal[i], \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        input_gen1 = tf.stack(input_gen1, axis=1, name='input_gen1_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        input_gen2, gen2 = zip(*[generator2(inputs=gen1[i], encode=encode_unstack_normal[i], \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        input_gen2 = tf.stack(input_gen2, axis=1, name='input_gen2_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        input_gen3, gen3 = zip(*[generator3(inputs=gen2[i], encode=encode_unstack_normal[i], \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM, INPUT_LENGTH // 4]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2, 4, CLASS_NUM, INPUT_LENGTH // 4]
        input_gen3 = tf.stack(input_gen3, axis=1, name='input_gen3_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        input_gen4 = [generator4(inputs=gen3[i], encode=encode_unstack_normal[i], \
                                            num=i, train=train) for i in range(CHANNEL_NUM)]
        # shape: [CHANNEL_NUM, None, CLASS_NUM, INPUT_LENGTH]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2, CLASS_NUM, INPUT_LENGTH]
        input_gen4 = tf.stack(input_gen4, axis=1, name='input_gen4_stack')
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Generators set')
    with tf.name_scope('discriminator'):
        encode_normal = tf.distributions.Normal(loc=encode_mean, scale=encode_var)
        # shape: [None, CHANNEL_NUM, 4, 16]
        dis1_real = discriminator1_conditional(inputs=real_input_1, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        dis1_gen = discriminator1_conditional(inputs=input_gen1, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        dis2_real = discriminator2_conditional(inputs=real_input_2, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        dis2_gen = discriminator2_conditional(inputs=input_gen2, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        dis3_real = discriminator3_conditional(inputs=real_input_3, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        dis3_gen = discriminator3_conditional(inputs=input_gen3, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        dis4_real = discriminator4_conditional(inputs=real_input_4, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        dis4_gen = discriminator4_conditional(inputs=input_gen4, encode=encode_normal)
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Discriminators set')
    with tf.name_scope('loss'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real) + gradient_penalty(real=real_input_1, \
                                        gen=input_gen1, encode=encode_normal, discriminator=discriminator1_conditional)
        mean_gen1, dev_gen1 = tf.nn.moments(input_gen1, axes=list(range(2, input_gen1.shape.ndims)))
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real) + gradient_penalty(real=real_input_2, \
                                        gen=input_gen2, encode=encode_normal, discriminator=discriminator2_conditional)
        mean_gen2, dev_gen2 = tf.nn.moments(input_gen2, axes=list(range(2, input_gen2.shape.ndims)))
        loss_gen2 = -tf.reduce_mean(dis2_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen1, mean_gen2)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen1, dev_gen2))
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real) + gradient_penalty(real=real_input_3, \
                                        gen=input_gen3, encode=encode_normal, discriminator=discriminator3_conditional)
        mean_gen3, dev_gen3 = tf.nn.moments(input_gen3, axes=list(range(2, input_gen3.shape.ndims)))
        loss_gen3 = -tf.reduce_mean(dis3_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen2, mean_gen3)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen2, dev_gen3))
        loss_dis4 = tf.reduce_mean(dis4_gen - dis4_real) + gradient_penalty(real=real_input_4, \
                                        gen=input_gen4, encode=encode_normal, discriminator=discriminator4_conditional)
        mean_gen4, dev_gen4 = tf.nn.moments(input_gen4, axes=list(range(2, input_gen4.shape.ndims)))
        loss_gen4 = -tf.reduce_mean(dis4_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen3, mean_gen4)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen3, dev_gen4))
        loss_gen = (loss_gen1 + loss_gen2 + loss_gen3 + loss_gen4) / 4.0
        tf.summary.scalar('loss_dis1', loss_dis1)
        tf.summary.scalar('loss_gen1', loss_gen1)
        tf.summary.scalar('dis1_real', tf.reduce_mean(dis1_real))
        tf.summary.scalar('loss_dis2', loss_dis2)
        tf.summary.scalar('loss_gen2', loss_gen2)
        tf.summary.scalar('dis2_real', tf.reduce_mean(dis2_real))
        tf.summary.scalar('loss_dis3', loss_dis3)
        tf.summary.scalar('loss_gen3', loss_gen3)
        tf.summary.scalar('dis3_real', tf.reduce_mean(dis3_real))
        tf.summary.scalar('loss_dis4', loss_dis4)
        tf.summary.scalar('loss_gen4', loss_gen4)
        tf.summary.scalar('dis4_real', tf.reduce_mean(dis4_real))
        tf.summary.scalar('loss_gen', loss_gen)
    print('Losses set')
    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Noise_generator')
    for i in range(CHANNEL_NUM):
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Time_seq_noise_generator' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator1_' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator2_' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator3_' + str(i))
        gen_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator4_' + str(i))
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1_Conditional')
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2_Conditional')
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3_Conditional')
    dis4_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator4_Conditional')
    gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Noise_generator')
    for i in range(CHANNEL_NUM):
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Time_seq_noise_generator' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator1_' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator2_' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator3_' + str(i))
        gen_extra_update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator4_' + str(i))
    dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1_Conditional')
    dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2_Conditional')
    dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3_Conditional')
    dis4_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator4_Conditional')
    with tf.name_scope('optimizers'):
        with tf.control_dependencies(dis1_extra_update_ops):
            dis1_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99).minimize(\
                                                    loss=loss_dis1, var_list=dis1_var, name='dis1_train')
        print('dis1_train setup')
        with tf.control_dependencies(dis2_extra_update_ops):
            dis2_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99).minimize(\
                                                    loss=loss_dis2, var_list=dis2_var, name='dis2_train')
        print('dis2_train setup')
        with tf.control_dependencies(dis3_extra_update_ops):
            dis3_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99).minimize(\
                                                    loss=loss_dis3, var_list=dis3_var, name='dis3_train')
        print('dis3_train setup')
        with tf.control_dependencies(dis4_extra_update_ops):
            dis4_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99).minimize(\
                                                    loss=loss_dis4, var_list=dis4_var, name='dis4_train')
        print('dis4_train setup')
        with tf.control_dependencies(gen_extra_update_ops):
            gen_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99).minimize(\
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
        train_count = 0
        feed_dict = {input_noise1: None, input_noise2: None, input_noise3: None, input_noise4: None, train: True}
        print('preparing complete')
        if sampling:
            feed_dict = {input_noise1: None, input_noise2: None, input_noise3: None, \
                                    input_noise4: None, real_input_4: None, train: True}
            path = args.sample
            try:
                feed_dict[real_input_4] = roll(path)[:BATCH_SIZE]
            except:
                print('Error while opening file.')
                return
            feed_dict[input_noise1] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
            feed_dict[input_noise2] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
            feed_dict[input_noise3] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
            feed_dict[input_noise4] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
            feed_dict[train] = True
            samples = sess.run(input_gen4, feed_dict=feed_dict)
            path = path.split('/')[-1]
            if not os.path.exists('Samples/sample_%s'):
                os.mkdir('Samples/sample_%s' % path)
            np.save(file='Samples/sample_%s' % path + '/%s' % path, arr=samples)
            unpack_sample(name='Samples/sample_%s' % path+ '/%s.npy' % path, concat=args.concat)
            return
        writer = tf.summary.FileWriter('train', sess.graph)
        epoch_num = 1000000
        for ___ in tqdm(range(epoch_num)):
            #for path in tqdm(pathlist):
                #input_data = np.load(str(path))
                #sess.run(iterator.initializer, feed_dict={data: input_data})
                #for __ in tqdm(range(4 * TRAIN_RATIO_DIS + TRAIN_RATIO_DIS)):
            feed_dict[train] = True
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            for i in range(TRAIN_RATIO_DIS):
                feed_dict[input_noise1] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise2] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise3] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                feed_dict[input_noise4] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                _, loss_val_dis1 = sess.run([dis1_train, loss_dis1], feed_dict=feed_dict, options=run_options)
                _, loss_val_dis2 = sess.run([dis2_train, loss_dis2], feed_dict=feed_dict, options=run_options)
                _, loss_val_dis3 = sess.run([dis3_train, loss_dis3], feed_dict=feed_dict, options=run_options)
                _, loss_val_dis4 = sess.run([dis4_train, loss_dis4], feed_dict=feed_dict, options=run_options)
            for i in range(TRAIN_RATIO_GEN):
                feed_dict[input_noise1] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise2] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise3] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                feed_dict[input_noise4] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                summary, _, loss_val_gen = sess.run([merged, gen_train, loss_gen], \
                                                    feed_dict=feed_dict, options=run_options)
            writer.add_summary(summary, train_count)
            tqdm.write('%06d' % train_count, end=' ')
            tqdm.write('Discriminator1 loss : %.7f' % loss_val_dis1, end=' ')
            tqdm.write('Discriminator2 loss : %.7f' % loss_val_dis2, end=' ')
            tqdm.write('Discriminator3 loss : %.7f' % loss_val_dis3, end=' ')
            tqdm.write('Discriminator4 loss : %.7f' % loss_val_dis4, end=' ')
            tqdm.write('Generator loss : %.7f' % loss_val_gen)
            train_count += 1
            if train_count % 1000 == 1:
                feed_dict[input_noise1] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise2] = get_noise([BATCH_SIZE, 1, 1, NOISE_LENGTH])
                feed_dict[input_noise3] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                feed_dict[input_noise4] = get_noise([BATCH_SIZE, CHANNEL_NUM, 1, NOISE_LENGTH])
                samples = sess.run(input_gen4, feed_dict=feed_dict)
                np.save(file='Samples/song_%06d' % train_count, arr=samples)
                unpack_sample('Samples/song_%06d' % train_count)
                save_path = saver.save(sess, 'Checkpoints/song_%06d' % train_count + '.ckpt')
                tqdm.write('Model Saved: %s' % save_path)
        writer.close()
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
