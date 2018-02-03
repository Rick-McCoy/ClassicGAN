from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pathlib
import os
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_NUM
from Model import get_noise, generator1, generator2, generator3, noise_generator, time_seq_noise_generator, discriminator1, discriminator2, discriminator3
#import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic
# checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    NOISE_LENGTH = 64
    TOTAL_TRAIN_EPOCH = 100
    LAMBDA = 10
    TRAIN_RATIO = 5

    with tf.name_scope('input_noises'):
        input_noise1 = tf.placeholder(dtype=tf.float32, shape=[NOISE_LENGTH], name='input_noise1')
        input_noise2 = tf.placeholder(dtype=tf.float32, shape=[1, NOISE_LENGTH], name='input_noise2')
        input_noise3 = tf.placeholder(dtype=tf.float32, shape=[CHANNEL_NUM, NOISE_LENGTH], name='input_noise3')
        input_noise4 = tf.placeholder(dtype=tf.float32, shape=[1, NOISE_LENGTH], name='input_noise4')
        train = tf.placeholder(dtype=tf.bool, name='traintest')
        noise1 = tf.stack(values=[input_noise1] * BATCH_NUM, axis=0)
        noise1 = tf.stack(values=[noise1] * CHANNEL_NUM, axis=0, name='noise1')
        noise2 = noise_generator(noise=input_noise2)
        noise2 = tf.stack(values=[noise2] * CHANNEL_NUM, axis=0, name='noise2')
        noise3 = tf.stack(values=[input_noise3] * BATCH_NUM, axis=1, name='noise3')
        noise4 = tf.stack(values=[time_seq_noise_generator(noise=input_noise4, num=i) for i in range(CHANNEL_NUM)], axis=0, name='noise4')
        input_noise = tf.concat(values=[noise1, noise2, noise3, noise4], axis=2, name='input_noise')
    with tf.name_scope('input_real'):
        real_input_3 = tf.placeholder(dtype=tf.float32, shape=[BATCH_NUM, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH], name='real_input_3')
        real_input_2 = tf.layers.max_pooling2d(inputs=real_input_3, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='real_input_2')
        real_input_1 = tf.layers.max_pooling2d(inputs=real_input_2, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='real_input_1')
        for i in range(CHANNEL_NUM):
            tf.summary.image('input_real' + str(i), tf.transpose(real_input_3[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
            tf.summary.image('input_real_1_' + str(i), tf.transpose(real_input_1[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
            tf.summary.image('input_real_2_' + str(i), tf.transpose(real_input_2[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
    print('Inputs set')
    with tf.name_scope('generator1'):
        gen1 = [generator1(noise=input_noise[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen1 = tf.stack(gen1, axis=1, name='gen1_stack')
    print('Generator1 set')
    with tf.name_scope('discriminator1'):
        dis1_real = discriminator1(inputs=real_input_1)
        dis1_gen = discriminator1(inputs=input_gen1, reuse=True)
    print('Discriminator1 set')
    with tf.name_scope('loss1'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real)
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen1 - real_input_1
        interpolate = real_input_1 + alpha * diff
        gradients = tf.gradients(discriminator1(inputs=interpolate, reuse=True), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis1+=LAMBDA * gradient_penalty
        tf.summary.scalar('dis1_gen', tf.reduce_mean(dis1_gen))
        tf.summary.scalar('dis1_real', tf.reduce_mean(dis1_real))
        tf.summary.scalar('discriminator_loss1', loss_dis1)
        tf.summary.scalar('generator_loss1', loss_gen1)
    print('Loss1 set')
    with tf.name_scope('generator2'):
        gen2 = [generator2(inputs=input_gen1, noise=input_noise[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen2 = tf.stack(gen2, axis=1, name='gen2_stack')
    print('Generator2 set')
    with tf.name_scope('discriminator2'):
        dis2_real = discriminator2(inputs=real_input_2)
        dis2_gen = discriminator2(inputs=input_gen2, reuse=True)
    print('Discriminator2 set')
    with tf.name_scope('loss2'):
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real)
        loss_gen2 = -tf.reduce_mean(dis2_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen2 - real_input_2
        interpolate = real_input_2 + alpha * diff
        gradients = tf.gradients(ys=discriminator2(inputs=interpolate, reuse=True), xs=interpolate)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis2+=LAMBDA * gradient_penalty
        tf.summary.scalar('dis2_gen', tf.reduce_mean(dis2_gen))
        tf.summary.scalar('dis2_real', tf.reduce_mean(dis2_real))
        tf.summary.scalar('discriminator_loss2', loss_dis2)
        tf.summary.scalar('generator_loss2', loss_gen2)
    print('Loss2 set')
    with tf.name_scope('generator3'):
        gen3 = [generator3(inputs=input_gen2, noise=input_noise[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen3 = tf.stack(gen3, axis=1, name='gen3_stack')
    print('Generator3 set')
    with tf.name_scope('discriminator3'):
        dis3_real = discriminator3(inputs=real_input_3)
        dis3_gen = discriminator3(inputs=input_gen3, reuse=True)
    print('Discriminator3 set')
    with tf.name_scope('loss3'):
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real)
        loss_gen3 = -tf.reduce_mean(dis3_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen3 - real_input_3
        interpolate = real_input_3 + alpha * diff
        gradients = tf.gradients(discriminator3(inputs=interpolate, reuse=True), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis3+=LAMBDA * gradient_penalty
        tf.summary.scalar('dis3_gen', tf.reduce_mean(dis3_gen))
        tf.summary.scalar('dis3_real', tf.reduce_mean(dis3_real))
        tf.summary.scalar('discriminator_loss3', loss_dis3)
        tf.summary.scalar('generator_loss3', loss_gen3)
    print('Loss3 set')
    noise_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Noise_generator')
    time_seq_noise_gen_var = []
    for i in range(CHANNEL_NUM):
        time_seq_noise_gen_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Time_seq_noise_generator' + str(i))
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')
    gen1_var = noise_gen_var + time_seq_noise_gen_var
    for i in range(CHANNEL_NUM):
        gen1_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator1_' + str(i))
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2')
    gen2_var = []
    for i in range(CHANNEL_NUM):
        gen2_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator2_' + str(i))
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3')
    gen3_var = []
    for i in range(CHANNEL_NUM):
        gen3_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator3_' + str(i))
    noise_gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Noise_generator')
    time_seq_noise_gen_extra_update_ops = []
    for i in range(CHANNEL_NUM):
        time_seq_noise_gen_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Time_seq_noise_generator' + str(i))
    dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1')
    gen1_extra_update_ops = noise_gen_extra_update_ops + time_seq_noise_gen_extra_update_ops
    for i in range(CHANNEL_NUM):
        gen1_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator1_' + str(i))
    dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2')
    gen2_extra_update_ops = []
    for i in range(CHANNEL_NUM):
        gen2_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator2_' + str(i))
    dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3')
    gen3_extra_update_ops = []
    for i in range(CHANNEL_NUM):
        gen3_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator3_' + str(i))
    with tf.name_scope('optimizers'):
        with tf.control_dependencies(dis1_extra_update_ops):
            dis1_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis1, var_list=dis1_var, name='dis1_train')
        print('dis1_train setup')
        with tf.control_dependencies(gen1_extra_update_ops):
            gen1_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen1, var_list=gen1_var, name='gen1_train')
        print('gen1_train setup')
        with tf.control_dependencies(dis2_extra_update_ops):
            dis2_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis2, var_list=dis2_var, name='dis2_train')
        print('dis2_train setup')
        with tf.control_dependencies(gen2_extra_update_ops):
            gen2_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen2, var_list=gen2_var, name='gen2_train')
        print('gen2_train setup')
        with tf.control_dependencies(dis3_extra_update_ops):
            dis3_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis3, var_list=dis3_var, name='dis3_train')
        print('dis3_train setup')
        with tf.control_dependencies(gen3_extra_update_ops):
            gen3_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen3, var_list=gen3_var, name='gen3_train')
        print('gen3_train setup')
    print('Optimizers set')
    loss_val_dis1 = loss_val_dis2 = loss_val_dis3 = loss_val_gen1 = loss_val_gen2 = loss_val_gen3 = 1.0
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allocator_type = 'BFC'
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train', sess.graph)
        sess.run(tf.global_variables_initializer())
        if tf.train.latest_checkpoint('Checkpoints') is not None:
            saver.restore(sess, tf.train.latest_checkpoint('Checkpoints'))
        pathlist = list(pathlib.Path('Classics').glob('**/*.mid')) + list(pathlib.Path('TPD').glob('**/*.mid'))
        train_count = 0
        print('preparing complete')
        for epoch in tqdm(range(TOTAL_TRAIN_EPOCH)):
            random.shuffle(pathlist)
            for songnum, path in enumerate(tqdm(pathlist)):
                try:
                    batch_input = roll(path)
                except Exception:
                    continue
                tqdm.write(str(path))
                feed_noise1 = get_noise([NOISE_LENGTH])
                feed_noise2 = get_noise([1, NOISE_LENGTH])
                feed_noise3 = get_noise([CHANNEL_NUM, NOISE_LENGTH])
                feed_noise4 = get_noise([1, NOISE_LENGTH])
                feed_dict = {input_noise1: feed_noise1, input_noise2: feed_noise2, input_noise3: feed_noise3, input_noise4: feed_noise4, real_input_3: batch_input, train: True}
                summary, _, loss_val_dis1 = sess.run([merged, dis1_train, loss_dis1], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                summary, _, loss_val_dis2 = sess.run([merged, dis2_train, loss_dis2], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                summary, _, loss_val_dis3 = sess.run([merged, dis3_train, loss_dis3], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                for i in range(TRAIN_RATIO):
                    summary, _, loss_val_gen1 = sess.run([merged, gen1_train, loss_gen1], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator1 loss: {:.7}'.format(loss_val_dis1) + ' Generator1 loss: {:.7}'.format(loss_val_gen1))
                for i in range(TRAIN_RATIO):
                    summary, _, loss_val_gen2 = sess.run([merged, gen2_train, loss_gen2], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator2 loss: {:.7}'.format(loss_val_dis2) + ' Generator2 loss: {:.7}'.format(loss_val_gen2))
                for i in range(TRAIN_RATIO):
                    summary, _, loss_val_gen3 = sess.run([merged, gen3_train, loss_gen3], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator3 loss: {:.7}'.format(loss_val_dis3) + ' Generator3 loss: {:.7}'.format(loss_val_gen3))
                if songnum % 1000 == 0:
                    samples = sess.run([gen3], feed_dict=feed_dict)
                    np.save(file='Samples/song_%06d' % (songnum + epoch * len(pathlist)), arr=samples)
                    save_path = saver.save(sess, 'Checkpoints/song_%06d' % (songnum + epoch * len(pathlist)) + '.ckpt')
                    tqdm.write('Model Saved: %s' % save_path)
        writer.close()
if __name__ == '__main__':
    main()
