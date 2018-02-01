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

    #SHARED_NOISE_LENGTH = 1000
    NOISE_LENGTH = 64
    TOTAL_TRAIN_EPOCH = 100
    LAMBDA = 10

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
        real_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_NUM, CLASS_NUM, INPUT_LENGTH, CHANNEL_NUM], name='real_input')
        real_input_2 = tf.layers.max_pooling2d(inputs=real_input, pool_size=[2, 2], strides=2, padding='same', name='real_input_2')
        real_input_1 = tf.layers.max_pooling2d(inputs=real_input_2, pool_size=[2, 2], strides=2, padding='same', name='real_input_1')
        for i in range(CHANNEL_NUM):
            tf.summary.image('input_real' + str(i), real_input[:, :, :, i:i + 1])
    print('Inputs set')
    with tf.name_scope('generator1'):
        gen1 = [generator1(noise=input_noise[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen1 = tf.stack(gen1, axis=3, name='gen1_stack')
    print('Generator1 set')
    with tf.name_scope('discriminator1'):
        dis1_real = discriminator1(inputs=real_input_1)
        dis1_gen = discriminator1(inputs=input_gen1, reuse=True)
    print('Discriminator1 set')
    with tf.name_scope('loss1'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real)
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        alpha = tf.random_uniform(shape=tf.shape(real_input_1), minval=0., maxval=1.)
        diff = input_gen1 - real_input_1
        interpolate = real_input_1 + tf.multiply(diff, alpha)
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
        gen2 = [generator2(inputs=gen1[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen2 = tf.stack(gen2, axis=3, name='gen2_stack')
    print('Generator2 set')
    with tf.name_scope('discriminator2'):
        dis2_real = discriminator2(inputs=real_input_2)
        dis2_gen = discriminator2(inputs=input_gen2, reuse=True)
    print('Discriminator2 set')
    with tf.name_scope('loss2'):
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real)
        loss_gen2 = -tf.reduce_mean(dis2_gen)
        alpha = tf.random_uniform(shape=tf.shape(real_input_2), minval=0., maxval=1.)
        diff = input_gen2 - real_input_2
        interpolate = real_input_2 + tf.multiply(diff, alpha)
        gradients = tf.gradients(discriminator2(inputs=interpolate, reuse=True), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis2+=LAMBDA * gradient_penalty
        tf.summary.scalar('dis2_gen', tf.reduce_mean(dis2_gen))
        tf.summary.scalar('dis2_real', tf.reduce_mean(dis2_real))
        tf.summary.scalar('discriminator_loss2', loss_dis2)
        tf.summary.scalar('generator_loss2', loss_gen2)
    print('Loss2 set')
    with tf.name_scope('generator3'):
        gen3 = [generator3(inputs=gen2[i], num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen3 = tf.stack(gen3, axis=3, name='gen3_stack')
    print('Generator3 set')
    with tf.name_scope('discriminator3'):
        dis3_real = discriminator3(inputs=real_input)
        dis3_gen = discriminator3(inputs=input_gen3, reuse=True)
    print('Discriminator3 set')
    with tf.name_scope('loss3'):
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real)
        loss_gen3 = -tf.reduce_mean(dis3_gen)
        alpha = tf.random_uniform(shape=tf.shape(real_input), minval=0., maxval=1.)
        diff = input_gen3 - real_input
        interpolate = real_input + tf.multiply(diff, alpha)
        gradients = tf.gradients(discriminator3(inputs=interpolate, reuse=True), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis3+=LAMBDA * gradient_penalty
        tf.summary.scalar('dis3_gen', tf.reduce_mean(dis3_gen))
        tf.summary.scalar('dis3_real', tf.reduce_mean(dis3_real))
        tf.summary.scalar('discriminator_loss3', loss_dis3)
        tf.summary.scalar('generator_loss3', loss_gen3)
    print('Loss3 set')
    noise_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='noise2')
    time_seq_noise_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='noise4')
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1')
    gen1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator1') + noise_gen_var + time_seq_noise_gen_var
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator2')
    gen2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator2')
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator3')
    gen3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator3')
    with tf.name_scope('optimizers'):
        noise_gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='noise2')
        time_seq_noise_gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='noise4')
        dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator1')
        gen1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator1') + noise_gen_extra_update_ops + time_seq_noise_gen_extra_update_ops
        dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator2')
        gen2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator2')
        dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator3')
        gen3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator3')
        with tf.control_dependencies(dis1_extra_update_ops):
            dis1_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis1, var_list=dis1_var, name='dis1_train')
        with tf.control_dependencies(gen1_extra_update_ops):
            gen1_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen1, var_list=dis1_var, name='gen1_train')
        with tf.control_dependencies(dis2_extra_update_ops):
            dis2_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis2, var_list=dis2_var, name='dis2_train')
        with tf.control_dependencies(gen2_extra_update_ops):
            gen2_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen2, var_list=dis2_var, name='gen2_train')
        with tf.control_dependencies(dis3_extra_update_ops):
            dis3_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis3, var_list=dis3_var, name='dis3_train')
        with tf.control_dependencies(gen3_extra_update_ops):
            gen3_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen3, var_list=dis3_var, name='gen3_train')
    print('Optimizers set')
    #with tf.name_scope('inputs'):
    #    inputs = tf.placeholder(tf.float32, [None, CLASS_NUM, INPUT_LENGTH,
    #    CHANNEL_NUM], name='inputs')
    #    shared_noise = tf.placeholder(tf.float32, [None, SHARED_NOISE_LENGTH],
    #    name='shared_noise')
    #    noise = tf.placeholder(tf.float32, [CHANNEL_NUM, None, NOISE_LENGTH],
    #    name='noise')
    #    train = tf.placeholder(tf.bool, name='traintest')
    #    for i in range(CHANNEL_NUM):
    #        tf.summary.image('input'+str(i), inputs[:, :, :, i:i+1])
    #with tf.name_scope('gen'):
    #    gen = [generator(noise[i], shared_noise, i, train) for i in
    #    range(CHANNEL_NUM)]
    #    input_gen = tf.stack(gen, axis=3, name='gen_stack')
    #print('generator set')
    #with tf.name_scope('dis'):
    #    dis_real = discriminator(inputs=inputs)
    #    dis_gene = discriminator(inputs=input_gen, reuse=True)
    #print('discriminator set')
    #with tf.name_scope('loss'):
    #    loss_dis = tf.reduce_mean(dis_gene - dis_real)
    #    loss_gen = -tf.reduce_mean(dis_gene)
    #    alpha = tf.random_uniform(shape=tf.shape(inputs), minval=0.,
    #    maxval=1.)
    #    diff = input_gen - inputs
    #    interpolate = inputs + tf.multiply(diff, alpha)
    #    gradients = tf.gradients(discriminator(inputs=interpolate,
    #    reuse=True), [interpolate])[0]
    #    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
    #    reduction_indices=[1]))
    #    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    #    loss_dis+=LAMBDA * gradient_penalty
    #    tf.summary.scalar('discriminator_loss', loss_dis)
    #    tf.summary.scalar('generator_loss', loss_gen)
    #print('loss set')
    #dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
    #gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
    #with tf.name_scope('optimizers'):
    #    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #    with tf.control_dependencies(extra_update_ops):
    #        gen_train = tf.train.AdamOptimizer(learning_rate=0.0001,
    #        beta1=0.5, beta2=0.9).minimize(loss_gen, var_list=gen_var,
    #        name='gen_train')
    #    dis_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5,
    #    beta2=0.9).minimize(loss_dis, var_list=dis_var, name='dis_train')
    #print('optimizer set')
    #loss_val_dis = loss_val_gen = 1.0
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
                feed_dict = {input_noise1: feed_noise1, input_noise2: feed_noise2, input_noise3: feed_noise3, input_noise4: feed_noise4, real_input: batch_input, train: True}
                summary, _, loss_val_dis1 = sess.run([merged, dis1_train, loss_dis1], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                for i in range(5):
                    summary, _, loss_val_gen1 = sess.run([merged, gen1_train, loss_gen1], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator1 loss: {:.7}'.format(loss_val_dis1) + ' Generator1 loss: {:.7}'.format(loss_val_gen1))
                summary, _, loss_val_dis2 = sess.run([merged, dis2_train, loss_dis2], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                for i in range(5):
                    summary, _, loss_val_gen2 = sess.run([merged, gen2_train, loss_gen2], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator2 loss: {:.7}'.format(loss_val_dis2) + ' Generator2 loss: {:.7}'.format(loss_val_gen2))
                summary, _, loss_val_dis3 = sess.run([merged, dis3_train, loss_dis3], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                for i in range(5):
                    summary, _, loss_val_gen3 = sess.run([merged, gen3_train, loss_gen3], feed_dict=feed_dict)
                    writer.add_summary(summary, train_count)
                    train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator3 loss: {:.7}'.format(loss_val_dis3) + ' Generator3 loss: {:.7}'.format(loss_val_gen3))
                if songnum % 500 == 0:
                    samples = sess.run([gen3], feed_dict=feed_dict)
                    np.save(file='Samples/song_%06d' % (songnum + epoch * len(pathlist)), arr=samples)
                    save_path = saver.save(sess, 'Checkpoints/song_%06d' % (songnum + epoch * len(pathlist)) + '.ckpt')
                    tqdm.write('Model Saved: %s' % save_path)
                #feed_shared_noise = get_noise([batch_num,
                #SHARED_NOISE_LENGTH])
                #feed_noise = get_noise([CHANNEL_NUM, batch_num, NOISE_LENGTH])
                #summary, _, loss_val_dis = sess.run([merged, dis_train,
                #loss_dis], feed_dict={inputs: batch_input, noise: feed_noise,
                #shared_noise: feed_shared_noise, train: True})
                #writer.add_summary(summary, train_count)
                #train_count += 1
                #for i in range(5):
                #    summary, _, loss_val_gen = sess.run([merged, gen_train,
                #    loss_gen], feed_dict={inputs: batch_input, noise:
                #    feed_noise, shared_noise: feed_shared_noise, train: True})
                #    writer.add_summary(summary, train_count)
                #    train_count += 1
                #tqdm.write('%06d' % train_count + ' Discriminator loss:
                #{:.7}'.format(loss_val_dis) + ' Generator loss:
                #{:.7}'.format(loss_val_gen))
                #if songnum % 500 == 0:
                #    batch_num = 15
                #    feed_shared_noise = get_noise([batch_num,
                #    SHARED_NOISE_LENGTH])
                #    feed_noise = get_noise([CHANNEL_NUM, batch_num,
                #    NOISE_LENGTH])
                #    samples = sess.run([gen], feed_dict={noise: feed_noise,
                #    shared_noise: feed_shared_noise, train: False})
                #    samples = np.stack(samples)
                #    np.save(file='Samples/song_%06d' % (songnum + epoch *
                #    len(pathlist)), arr=samples)
                #    save_path = saver.save(sess, 'Checkpoints/song_%06d' %
                #    (songnum + epoch * len(pathlist)) + '.ckpt')
                #    tqdm.write('Model Saved: %s' % save_path)
        writer.close()
if __name__ == '__main__':
    main()
