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
from Data import roll, channel_num, class_num, input_length
from Model import generator, discriminator, get_noise
#import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic
# checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    shared_noise_len = 400
    noise_length = 200
    total_train_epoch = 100
    Lambda = 10

    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, [None, class_num, input_length, channel_num], name='inputs')
        sharednoise = tf.placeholder(tf.float32, [None, shared_noise_len], name='sharednoise')
        noise = tf.placeholder(tf.float32, [channel_num, None, noise_length], name='noise')
        train = tf.placeholder(tf.bool, name='traintest')
    with tf.name_scope('gen'):
        gen = [0] * channel_num
        for i in range(channel_num):
            gen[i] = generator(noise, sharednoise, i, train)
        input_gen = tf.stack(gen, axis=3, name='gen_stack')
    print('generator set')
    with tf.name_scope('dis'):
        dis_real = discriminator(inputs=inputs)
        dis_gene = discriminator(inputs=input_gen, reuse=True)
    print('discriminator set')
    with tf.name_scope('loss'):
        #loss_dis = -tf.reduce_mean(tf.log(dis_real) + tf.log(1 - dis_gene))
        #loss_gen = -tf.reduce_mean(tf.log(dis_gene))
        loss_dis = tf.reduce_mean(dis_gene - dis_real)
        loss_gen = -tf.reduce_mean(dis_gene)
        alpha = tf.random_uniform(shape=tf.shape(inputs), minval=0., maxval=1.)
        diff = input_gen - inputs
        interpolate = inputs + tf.multiply(diff, alpha)
        gradients = tf.gradients(discriminator(inputs=interpolate, reuse=True), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_dis+=Lambda * gradient_penalty
        tf.summary.scalar('discriminator_loss', loss_dis)
        tf.summary.scalar('generator_loss', loss_gen)
    print('loss set')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
    with tf.name_scope('optimizers'):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            gen_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss_gen, var_list=gen_var, name='gen_train')
        dis_train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss_dis, var_list=dis_var, name='dis_train')
    print('optimizer set')
    loss_val_dis = loss_val_gen = 1.0
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train', sess.graph)
        sess.run(tf.global_variables_initializer())
        data = []
        pathlist = list(pathlib.Path('Classics').glob('**/*.mid')) + list(pathlib.Path('TPD').glob('**/*.mid'))
        train_count = 0
        print('preparing complete')
        for epoch in tqdm(range(total_train_epoch)):
            random.shuffle(pathlist)
            for songnum, path in enumerate(tqdm(pathlist)):
                try:
                    length, data = roll(path)
                except:
                    continue
                tqdm.write(str(path))
                n_batch = length // input_length
                batch_input = np.empty([n_batch, class_num, input_length, channel_num])
                for i in range(n_batch):
                    batch_input[i] = data[:, i * input_length:(i + 1) * input_length, :]
                feed_sharednoise = get_noise(n_batch, shared_noise_len)
                feed_noise = np.empty([channel_num, n_batch, noise_length])
                for i in range(channel_num):
                    feed_noise[i] = get_noise(n_batch, noise_length)
                summary, _, loss_val_dis = sess.run([merged, dis_train, loss_dis], feed_dict={inputs: batch_input, noise: feed_noise, sharednoise: feed_sharednoise, train: True})
                writer.add_summary(summary, train_count)
                train_count += 1
                for i in range(10):
                    summary, _, loss_val_gen = sess.run([merged, gen_train, loss_gen], feed_dict={inputs: batch_input, noise: feed_noise, sharednoise: feed_sharednoise, train: True})
                    writer.add_summary(summary, train_count)
                    train_count += 1
                tqdm.write('%06d' % train_count + ' D loss: {:.7}'.format(loss_val_dis) + ' G loss: {:.7}'.format(loss_val_gen))
                if songnum % 1000 == 0:
                    n_batch = 15
                    feed_sharednoise = get_noise(n_batch, shared_noise_len)
                    feed_noise = np.empty([channel_num, n_batch, noise_length])
                    for i in range(channel_num):
                        feed_noise[i] = get_noise(n_batch, noise_length)
                    samples = sess.run([gen], feed_dict={noise: feed_noise, sharednoise: feed_sharednoise, train: False})
                    samples = np.stack(samples)
                    np.save(file='Samples/song_%06d' % (songnum + epoch * len(pathlist)), arr=samples)
        writer.close()
if __name__ == '__main__':
    main()
