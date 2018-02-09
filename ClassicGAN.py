from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pathlib
import os
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_NUM
from Model import get_noise, generator1, generator2, generator3, noise_generator, time_seq_noise_generator, discriminator1_conditional, discriminator2_conditional, discriminator3_conditional, encoder, NOISE_LENGTH
import memory_saving_gradients
# monkey patch memory_saving_gradients.gradients_speed to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    TOTAL_TRAIN_EPOCH = 100
    LAMBDA = 10
    TRAIN_RATIO_DIS = 5
    TRAIN_RATIO_GEN = 1

    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('Samples'):
        os.makedirs('Samples')

    with tf.name_scope('inputs'):
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
        real_input_3 = tf.placeholder(dtype=tf.float32, shape=[BATCH_NUM, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH], name='real_input_3')
        encode = encoder(inputs=real_input_3, train=train)
        input_noise = tf.concat(values=[noise1, noise2, noise3, noise4], axis=2, name='input_noise')
        real_input_2 = tf.layers.max_pooling2d(inputs=real_input_3, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='real_input_2')
        real_input_1 = tf.layers.max_pooling2d(inputs=real_input_2, pool_size=[2, 2], strides=2, padding='same', data_format='channels_first', name='real_input_1')
        for i in range(CHANNEL_NUM):
            tf.summary.image('input_real' + str(i), tf.transpose(real_input_3[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
            tf.summary.image('input_real_1_' + str(i), tf.transpose(real_input_1[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
            tf.summary.image('input_real_2_' + str(i), tf.transpose(real_input_2[:BATCH_NUM // 10, i:i + 1, :, :], [0, 2, 3, 1]))
    print('Inputs set')
    with tf.name_scope('generator1'):
        input_gen1, gen1 = zip(*[generator1(noise=input_noise[i], encode=encode[i], num=i, train=train) for i in range(CHANNEL_NUM)])
        gen1 = tf.concat(gen1, axis=1, name='gen1_concat')
        input_gen1 = tf.stack(input_gen1, axis=1, name='input_gen1_stack')
    print('Generator1 set')
    with tf.name_scope('discriminator1'):
        dis1_real = discriminator1_conditional(inputs=real_input_1, encode=encode, train=train)
        dis1_gen = discriminator1_conditional(inputs=input_gen1, encode=encode, train=train)
    print('Discriminator1 set')
    with tf.name_scope('loss1'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real)
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen1 - real_input_1
        interpolate = real_input_1 + alpha * diff
        gradients = tf.gradients(discriminator1_conditional(inputs=interpolate, encode=encode, train=train), [interpolate])[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis1 += LAMBDA * gradient_penalty
        tf.summary.scalar('dis1_gen', tf.reduce_mean(dis1_gen))
        tf.summary.scalar('dis1_real', tf.reduce_mean(dis1_real))
        tf.summary.scalar('discriminator_loss1', loss_dis1)
        tf.summary.scalar('generator_loss1', loss_gen1)
    print('Loss1 set')
    with tf.name_scope('generator2'):
        input_gen2, gen2 = zip(*[generator2(inputs=gen1, encode=encode, num=i, train=train) for i in range(CHANNEL_NUM)])
        gen2 = tf.concat(gen2, axis=1, name='gen2_concat')
        input_gen2 = tf.stack(input_gen2, axis=1, name='input_gen2_stack')
    print('Generator2 set')
    with tf.name_scope('discriminator2'):
        dis2_real = discriminator2_conditional(inputs=real_input_2, encode=encode, train=train)
        dis2_gen = discriminator2_conditional(inputs=input_gen2, encode=encode, train=train)
    print('Discriminator2 set')
    with tf.name_scope('loss2'):
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real)
        loss_gen2 = -tf.reduce_mean(dis2_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen2 - real_input_2
        interpolate = real_input_2 + alpha * diff
        gradients = tf.gradients(ys=discriminator2_conditional(inputs=interpolate, encode=encode, train=train), xs=interpolate)[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis2 += LAMBDA * gradient_penalty
        tf.summary.scalar('dis2_gen', tf.reduce_mean(dis2_gen))
        tf.summary.scalar('dis2_real', tf.reduce_mean(dis2_real))
        tf.summary.scalar('discriminator_loss2', loss_dis2)
        tf.summary.scalar('generator_loss2', loss_gen2)
    print('Loss2 set')
    with tf.name_scope('generator3'):
        gen3 = [generator3(inputs=gen2, encode=encode, num=i, train=train) for i in range(CHANNEL_NUM)]
        input_gen3 = tf.stack(gen3, axis=1, name='gen3_stack')
    print('Generator3 set')
    with tf.name_scope('discriminator3'):
        dis3_real = discriminator3_conditional(inputs=real_input_3, encode=encode, train=train)
        dis3_gen = discriminator3_conditional(inputs=input_gen3, encode=encode, train=train)
    print('Discriminator3 set')
    with tf.name_scope('loss3'):
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real)
        loss_gen3 = -tf.reduce_mean(dis3_gen)
        alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
        diff = input_gen3 - real_input_3
        interpolate = real_input_3 + alpha * diff
        gradients = tf.gradients(discriminator3_conditional(inputs=interpolate, encode=encode, train=train), [interpolate])[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis3 += LAMBDA * gradient_penalty
        tf.summary.scalar('dis3_gen', tf.reduce_mean(dis3_gen))
        tf.summary.scalar('dis3_real', tf.reduce_mean(dis3_real))
        tf.summary.scalar('discriminator_loss3', loss_dis3)
        tf.summary.scalar('generator_loss3', loss_gen3)
    print('Loss3 set')
    noise_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Noise_generator')
    time_seq_noise_gen_var = []
    for i in range(CHANNEL_NUM):
        time_seq_noise_gen_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Time_seq_noise_generator' + str(i))
    encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    dis1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1_Conditional')
    gen1_var = noise_gen_var + time_seq_noise_gen_var + encoder_var
    for i in range(CHANNEL_NUM):
        gen1_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator1_' + str(i))
    dis2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2_Conditional')
    gen2_var = encoder_var
    for i in range(CHANNEL_NUM):
        gen2_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator2_' + str(i))
    dis3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3_Conditional')
    gen3_var = encoder_var
    for i in range(CHANNEL_NUM):
        gen3_var+=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator3_' + str(i))
    noise_gen_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Noise_generator')
    time_seq_noise_gen_extra_update_ops = []
    for i in range(CHANNEL_NUM):
        time_seq_noise_gen_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Time_seq_noise_generator' + str(i))
    encoder_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')
    dis1_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator1_Conditional')
    gen1_extra_update_ops = noise_gen_extra_update_ops + time_seq_noise_gen_extra_update_ops + encoder_extra_update_ops
    for i in range(CHANNEL_NUM):
        gen1_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator1_' + str(i))
    dis2_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator2_Conditional')
    gen2_extra_update_ops = encoder_extra_update_ops
    for i in range(CHANNEL_NUM):
        gen2_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator2_' + str(i))
    dis3_extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator3_Conditional')
    gen3_extra_update_ops = encoder_extra_update_ops
    for i in range(CHANNEL_NUM):
        gen3_extra_update_ops+=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator3_' + str(i))
    loss_dis = (loss_dis1 + loss_dis2 + loss_dis3) / 3.0
    dis_var = dis1_var + dis2_var + dis3_var
    dis_extra_update_ops = dis1_extra_update_ops + dis2_extra_update_ops + dis3_extra_update_ops
    loss_gen = (loss_gen1 + loss_gen2 + loss_gen3) / 3.0
    gen_var = gen1_var + gen2_var + gen3_var
    gen_extra_update_ops = gen1_extra_update_ops + gen2_extra_update_ops + gen3_extra_update_ops
    tf.summary.scalar('loss_dis', loss_dis)
    tf.summary.scalar('loss_gen', loss_gen)
    with tf.name_scope('optimizers'):
        with tf.control_dependencies(dis_extra_update_ops):
            dis_train = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(loss=loss_dis, var_list=dis_var, name='dis_train')
        print('dis_train setup')
        with tf.control_dependencies(gen_extra_update_ops):
            gen_train = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(loss=loss_gen, var_list=gen_var, name='gen_train')
        print('gen_train setup')
    print('Optimizers set')
    loss_val_dis1 = loss_val_dis2 = loss_val_dis3 = loss_val_gen1 = loss_val_gen2 = loss_val_gen3 = loss_val_dis = loss_val_gen = 1.0
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
        pathlist = list(pathlib.Path('Classics').glob('**/*.mid')) + list(pathlib.Path('TPD').glob('**/*.mid'))# + list(pathlib.Path('Lakh').glob('**/*.mid'))
        train_count = 0
        print('preparing complete')
        for epoch in tqdm(range(TOTAL_TRAIN_EPOCH)):
            random.shuffle(pathlist)
            for songnum, path in enumerate(tqdm(pathlist)):
                train_index = songnum // 300 % 3
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
                for i in range(TRAIN_RATIO_DIS):
                    _, loss_val_dis = sess.run([dis_train, loss_dis], feed_dict=feed_dict)
                for i in range(TRAIN_RATIO_GEN):
                    summary, _, loss_val_gen = sess.run([merged, gen_train, loss_gen], feed_dict=feed_dict)
                writer.add_summary(summary, train_count)
                train_count+=1
                tqdm.write('%06d' % train_count + ' Discriminator loss: {:.7}'.format(loss_val_dis) + ' Generator loss: {:.7}'.format(loss_val_gen))
                if train_count % 1000 == 0:
                    save_feed_dict = feed_dict
                    save_feed_dict[train] = False
                    samples = sess.run([gen3], feed_dict=save_feed_dict)
                    np.save(file='Samples/song_%06d' % train_count, arr=samples)
                    save_path = saver.save(sess, 'Checkpoints/song_%06d' % train_count + '.ckpt')
                    tqdm.write('Model Saved: %s' % save_path)
        writer.close()
if __name__ == '__main__':
    main()
