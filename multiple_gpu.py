from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_SIZE
from Model import get_noise, generator1, generator2, generator3, generator4, \
                    noise_generator, time_seq_noise_generator, discriminator1, \
                    discriminator2, discriminator3, discriminator4, encoder, \
                    NOISE_LENGTH
from Convert import unpack_sample
import memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_gpus = tf.contrib.eager.num_gpus()
LAMBDA = 10
LAMBDA1 = 1
LAMBDA2 = 5
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

def inference(real_input_4):

    with tf.name_scope('inputs'):
        input_noise1 = tf.random_normal(dtype=tf.float32, shape=[None, 1, 1, NOISE_LENGTH], name='input_noise1')
        input_noise2 = tf.random_normal(dtype=tf.float32, shape=[None, 1, 1, NOISE_LENGTH], name='input_noise2')
        input_noise3 = tf.random_normal(dtype=tf.float32, shape=[None, CHANNEL_NUM, 1, NOISE_LENGTH], name='input_noise3')
        input_noise4 = tf.random_normal(dtype=tf.float32, shape=[None, CHANNEL_NUM, 1, NOISE_LENGTH], name='input_noise4')

        train = tf.placeholder(dtype=tf.bool, name='traintest')
        noise1 = tf.tile(input=input_noise1, multiples=[1, CHANNEL_NUM, 4, 1], name='noise1')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        with tf.device('/cpu:0'):
            noise2_gen = noise_generator(noise=input_noise2, train=train)
        # shape: [None, 1, 4, NOISE_LENGTH]
        noise2 = tf.tile(input=noise2_gen, multiples=[1, CHANNEL_NUM, 1, 1], name='noise2')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        noise3 = tf.tile(input=input_noise3, multiples=[1, 1, 4, 1], name='noise3')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH]
        input_noise4_split = tf.split(input_noise4, num_or_size_splits=CHANNEL_NUM, axis=1)
        # shape: [CHANNEL_NUM, None, 1, NOISE_LENGTH]
        with tf.device('/cpu:0'):
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
        with tf.device('/cpu:0'):
            encode_split = [encoder(inputs=j, num=i, train=train) for i, j in enumerate(real_input_3_split)]
        # shape: [CHANNEL_NUM, None, 4, 16]
        encode = tf.stack(encode_split, axis=1, name='encode')
        # shape: [None, CHANNEL_NUM, 4, 16]

        input_noise = tf.concat(values=[noise1, noise2, noise3, noise4], axis=3, name='input_noise')
        # shape: [None, CHANNEL_NUM, 4, NOISE_LENGTH * 4]

        real_input_4_image = tf.expand_dims(real_input_4[:1], axis=-1, name='real_input_4_image_expand')
        # shape: [1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
        real_input_4_image = tf.unstack(real_input_4_image, axis=1, name='real_input_4_image_unstack')
        # shape: [CHANNEL_NUM, 1, CLASS_NUM, INPUT_LENGTH, 1]
        for i, j in enumerate(real_input_4_image):
            tf.summary.image('real_input_4_' + str(i), j)
        
        real_input_3_image = tf.unstack(real_input_3[:1], axis=2, name='real_input_3_image_unstack')
        # shape: [4, 1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH // 4]
        real_input_3_image = tf.concat(real_input_3_image, axis=-1, name='real_input_3_image_concat')
        # shape: [1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
        real_input_3_image = tf.expand_dims(real_input_3_image, axis=-1, name='real_input_3_expand')
        # shape: [1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
        real_input_3_image = tf.unstack(real_input_3_image, axis=1, name='real_input_3_image')
        # shape: [CHANNEL_NUM, 1, CLASS_NUM, INPUT_LENGTH, 1]
        for i, j in enumerate(real_input_3_image):
            tf.summary.image('real_input_3_' + str(i), j)
            
        real_input_2_image = tf.unstack(real_input_2[:1], axis=2, name='real_input_2_unstack')
        # shape: [4, 1, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 8]
        real_input_2_image = tf.concat(real_input_2_image, axis=-1, name='real_input_2_concat')
        # shape: [1, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2]
        real_input_2_image = tf.expand_dims(real_input_2_image, axis=-1, name='real_input_2_expand')
        # shape: [1, CHANNEL_NUM, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        real_input_2_image = tf.unstack(real_input_2_image, axis=1, name='real_input_2_image')
        # shape: [CHANNEL_NUM, 1, CLASS_NUM // 2, INPUT_LENGTH // 2, 1]
        for i, j in enumerate(real_input_2_image):
            tf.summary.image('real_input_2_' + str(i), j)
            
        real_input_1_image = tf.unstack(real_input_1[:1], axis=2, name='real_input_1_unstack')
        # shape: [4, 1, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 16]
        real_input_1_image = tf.concat(real_input_1_image, axis=-1, name='real_input_3_concat')
        # shape: [1, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4]
        real_input_1_image = tf.expand_dims(real_input_1_image, axis=-1, name='real_input_1_expand')
        # shape: [1, CHANNEL_NUM, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        real_input_1_image = tf.unstack(real_input_1_image, axis=1, name='real_input_1_image')
        # shape: [CHANNEL_NUM, 1, CLASS_NUM // 4, INPUT_LENGTH // 4, 1]
        for i, j in enumerate(real_input_1_image):
            tf.summary.image('real_input_1_' + str(i), j)

    with tf.name_scope('generator'):
        input_noise_split = tf.unstack(input_noise, axis=1, name='input_noise_split')
        # shape: [CHANNEL_NUM, None, 4, NOISE_LENGTH * 4]
        with tf.device('/cpu:0'):
            input_gen1, gen1 = zip(*[generator1(noise=input_noise_split[i], encode=encode[:, i], \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        input_gen1 = tf.stack(input_gen1, axis=1, name='input_gen1_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        gen1 = [tf.concat(values=[i, input_gen1], axis=1) for i in gen1]
        # shape: [CHANNEL_NUM, None, NOISE_LENGTH * 2 + CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        with tf.device('/cpu:0'):
            input_gen2, gen2 = zip(*[generator2(inputs=gen1[i], encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        # shape: [CHANNEL_NUM, None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        input_gen2 = tf.stack(input_gen2, axis=1, name='input_gen2_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
        gen2 = [tf.concat(values=[i, input_gen2], axis=1) for i in gen2]
        # shape: [CHANNEL_NUM, None, 32 + CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        with tf.device('/cpu:0'):
            input_gen3, gen3 = zip(*[generator3(inputs=gen2[i], encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
        # shape: [CHANNEL_NUM, None, 4, CLASS_NUM, INPUT_LENGTH // 4]
        # shape: [CHANNEL_NUM, None, 16, 4, CLASS_NUM, INPUT_LENGTH // 4]
        input_gen3 = tf.stack(input_gen3, axis=1, name='input_gen3_stack')
        # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
        gen3 = [tf.concat(values=[i, input_gen3], axis=1) for i in gen3]
        # shape: [CHANNEL_NUM, None, 16 + CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
        with tf.device('/cpu:0'):
            input_gen4 = [generator4(inputs=gen3[i], encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)]
        # shape: [CHANNEL_NUM, None, CLASS_NUM, INPUT_LENGTH]
        input_gen4 = tf.stack(input_gen4, axis=1, name='input_gen4_stack')
        # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Generators set')
    with tf.device('/cpu:0'):
        with tf.name_scope('discriminator'):
            dis1_real = discriminator1(inputs=real_input_1, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
            dis1_gen = discriminator1(inputs=input_gen1, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
            dis2_real = discriminator2(inputs=real_input_2, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
            dis2_gen = discriminator2(inputs=input_gen2, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
            dis3_real = discriminator3(inputs=real_input_3, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
            dis3_gen = discriminator3(inputs=input_gen3, encode=encode)
            # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
            dis4_real = discriminator4(inputs=real_input_4, encode=encode)
            # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
            dis4_gen = discriminator4(inputs=input_gen4, encode=encode)
            # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Discriminators set')
    with tf.name_scope('loss'):
        loss_dis1 = tf.reduce_mean(dis1_gen - dis1_real) + gradient_penalty(real=real_input_1, \
                                        gen=input_gen1, encode=encode, discriminator=discriminator1)
        mean_gen1, dev_gen1 = tf.nn.moments(input_gen1, axes=list(range(2, input_gen1.shape.ndims)))
        loss_gen1 = -tf.reduce_mean(dis1_gen)
        loss_dis2 = tf.reduce_mean(dis2_gen - dis2_real) + gradient_penalty(real=real_input_2, \
                                        gen=input_gen2, encode=encode, discriminator=discriminator2)
        mean_gen2, dev_gen2 = tf.nn.moments(input_gen2, axes=list(range(2, input_gen2.shape.ndims)))
        loss_gen2 = -tf.reduce_mean(dis2_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen1, mean_gen2)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen1, dev_gen2))
        loss_dis3 = tf.reduce_mean(dis3_gen - dis3_real) + gradient_penalty(real=real_input_3, \
                                        gen=input_gen3, encode=encode, discriminator=discriminator3)
        mean_gen3, dev_gen3 = tf.nn.moments(input_gen3, axes=list(range(2, input_gen3.shape.ndims)))
        loss_gen3 = -tf.reduce_mean(dis3_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen2, mean_gen3)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen2, dev_gen3))
        loss_dis4 = tf.reduce_mean(dis4_gen - dis4_real) + gradient_penalty(real=real_input_4, \
                                        gen=input_gen4, encode=encode, discriminator=discriminator4)
        mean_gen4, dev_gen4 = tf.nn.moments(input_gen4, axes=list(range(2, input_gen4.shape.ndims)))
        loss_gen4 = -tf.reduce_mean(dis4_gen) + LAMBDA1 * tf.reduce_mean(tf.squared_difference(mean_gen3, mean_gen4)) \
                                                + LAMBDA2 * tf.reduce_mean(tf.squared_difference(dev_gen3, dev_gen4))
        loss_gen = tf.add_n([loss_gen1, loss_gen2, loss_gen3, loss_gen4]) / 4
        tf.add_to_collection('loss_dis1', loss_dis1)
        tf.add_to_collection('loss_dis2', loss_dis2)
        tf.add_to_collection('loss_dis3', loss_dis3)
        tf.add_to_collection('loss_dis4', loss_dis4)
        tf.add_to_collection('loss_gen', loss_gen)

def make_iterator():
    filename = 'Dataset/dataset.tfrecord'
    dataset = tf.data.TFRecordDataset(filename)
    def _parse(example_proto):
        feature = {'roll' : tf.FixedLenFeature((6, 72, 384), tf.float32)}
        parsed = tf.parse_single_example(example_proto, feature)
        return parsed['roll']
    dataset = dataset.map(_parse, num_parallel_calls=8)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=20000))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    iterator = dataset.make_one_shot_iterator()
    return iterator

def tower_loss(scope, inputs):
    inference(inputs)

    loss_dis1 = tf.add_n(tf.get_collection('loss_dis1', scope=scope), name='sum_loss_dis1')
    loss_dis2 = tf.add_n(tf.get_collection('loss_dis2', scope=scope), name='sum_loss_dis2')
    loss_dis3 = tf.add_n(tf.get_collection('loss_dis3', scope=scope), name='sum_loss_dis3')
    loss_dis4 = tf.add_n(tf.get_collection('loss_dis4', scope=scope), name='sum_loss_dis4')
    loss_gen = tf.add_n(tf.get_collection('loss_gen', scope=scope), name='sum_loss_gen')
    
    tf.summary.scalar('loss_dis1', loss_dis1)
    tf.summary.scalar('loss_dis2', loss_dis2)
    tf.summary.scalar('loss_dis3', loss_dis3)
    tf.summary.scalar('loss_dis4', loss_dis4)
    tf.summary.scalar('loss_gen', loss_gen)

    return loss_dis1, loss_dis2, loss_dis3, loss_dis4, loss_gen

def average_gradients(tower_grads):
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grad = tf.stack([g for g, _ in grads_and_vars], axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        v = grads_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        dis1_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)
        dis2_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)
        dis3_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)
        dis4_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)
        gen_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)

        iterator = make_iterator()

        tower_grads_dis1 = []
        tower_grads_dis2 = []
        tower_grads_dis3 = []
        tower_grads_dis4 = []
        tower_grads_gen = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        inputs = iterator.get_next()

                        loss_dis1, loss_dis2, loss_dis3, loss_dis4, loss_gen = tower_loss(scope, inputs)
                        tf.get_variable_scope().reuse_variables()
                        grad_dis1 = dis1_opt.compute_gradients(loss_dis1)
                        grad_dis2 = dis2_opt.compute_gradients(loss_dis2)
                        grad_dis3 = dis3_opt.compute_gradients(loss_dis3)
                        grad_dis4 = dis4_opt.compute_gradients(loss_dis4)
                        grad_gen = gen_opt.compute_gradients(loss_gen)

                        tower_grads_dis1.append(grad_dis1)
                        tower_grads_dis1.append(grad_dis2)
                        tower_grads_dis1.append(grad_dis3)
                        tower_grads_dis1.append(grad_dis4)
                        tower_grads_dis1.append(grad_gen)

        grads_dis1 = average_gradients(tower_grads_dis1)
        grads_dis2 = average_gradients(tower_grads_dis2)
        grads_dis3 = average_gradients(tower_grads_dis3)
        grads_dis4 = average_gradients(tower_grads_dis4)
        grads_gen = average_gradients(tower_grads_gen)

        dis1_grad_op = dis1_opt.apply_gradients(grads_dis1)
        dis2_grad_op = dis1_opt.apply_gradients(grads_dis2)
        dis3_grad_op = dis1_opt.apply_gradients(grads_dis3)
        dis4_grad_op = dis1_opt.apply_gradients(grads_dis4)
        gen_grad_op = dis1_opt.apply_gradients(grads_gen)

        saver = tf.train.Saver(tf.global_variables())
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('train', sess.graph)
            epoch_num = 100000
            feed_dict = {train: True}
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            for train_count in tqdm(range(epoch_num)):
                feed_dict[train] = True
                for i in range(TRAIN_RATIO_DIS):
                    _, loss_val_dis1 = sess.run([dis1_grad_op, loss_dis1], feed_dict=feed_dict, options=run_options)
                    _, loss_val_dis2 = sess.run([dis2_grad_op, loss_dis2], feed_dict=feed_dict, options=run_options)
                    _, loss_val_dis3 = sess.run([dis3_grad_op, loss_dis3], feed_dict=feed_dict, options=run_options)
                    _, loss_val_dis4 = sess.run([dis4_grad_op, loss_dis4], feed_dict=feed_dict, options=run_options)

                for i in range(TRAIN_RATIO_GEN):
                    summary, _, loss_val_gen = sess.run([merged, gen_grad_op, loss_gen], feed_dict=feed_dict, options=run_options)
            
                writer.add_summary(summary, train_count)
                tqdm.write('%06d' % train_count, end=' ')
                tqdm.write('Discriminator1 loss : %.7f' % loss_val_dis1, end=' ')
                tqdm.write('Discriminator2 loss : %.7f' % loss_val_dis2, end=' ')
                tqdm.write('Discriminator3 loss : %.7f' % loss_val_dis3, end=' ')
                tqdm.write('Discriminator4 loss : %.7f' % loss_val_dis4, end=' ')
                tqdm.write('Generator loss : %.7f' % loss_val_gen)
                if train_count % 500 == 0:
                    save_path = saver.save(sess, 'Checkpoints/song_%06d' % train_count + '.ckpt')
                    tqdm.write('Model Saved: %s' % save_path)
                    
def main():

    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('Samples'):
        os.makedirs('Samples')

    train()

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()