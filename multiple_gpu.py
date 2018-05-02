from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import tensorflow as tf
import argparse
from tqdm import tqdm
from Data import roll, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, BATCH_SIZE
from Model import get_noise, generator1, generator2, generator3, \
                    discriminator1, discriminator2, discriminator3, \
                    shared_gen, encoder, NOISE_LENGTH
from tensorflow.contrib import eager, data
import memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_gpus = eager.num_gpus()
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

def inference(real_input_3):
    input_noise = tf.placeholder(dtype=tf.float32, shape=[None, NOISE_LENGTH, 4], name='input_noise')

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
    with tf.device('/cpu:0'):
        encode = encoder(inputs=real_input_2, train=train)
    # shape: [None, 4, 2, 8]

    real_input_3_image = tf.expand_dims(real_input_3[:1], axis=-1, name='real_input_3_image_expand')
    # shape: [1, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH, 1]
    real_input_2_image = tf.layers.max_pooling3d(inputs=real_input_3_image, pool_size=[2, 2, 1], \
                                                    strides=(2, 2, 1), padding='same', \
                                                    data_format='channels_first', \
                                                    name='real_input_2_image')
    real_input_1_image = tf.layers.max_pooling3d(inputs=real_input_2_image, pool_size=[2, 2, 1], \
                                                    strides=(2, 2, 1), padding='same', \
                                                    data_format='channels_first', \
                                                    name='real_input_1_image')
    for i in range(CHANNEL_NUM):
        tf.summary.image('real_input_1_' + str(i), real_input_1_image[:, i])
        tf.summary.image('real_input_2_' + str(i), real_input_2_image[:, i])
        tf.summary.image('real_input_3_' + str(i), real_input_3_image[:, i])

    with tf.device('/cpu:0'):
        shared_output = shared_gen(noise=input_noise, encode=encode, train=train)
    # shape: [None, 64, 4, CLASS_NUM // 4, INPUT_LENGTH // 16]
    with tf.device('/cpu:0'):
        output_gen1, gen1 = zip(*[generator1(inputs=shared_output, encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
    # shape: [CHANNEL_NUM, None, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
    # shape: [CHANNEL_NUM, None, 32, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
    output_gen1 = tf.stack(output_gen1, axis=1, name='output_gen1_stack')
    # shape: [None, CHANNEL_NUM, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
    gen1 = [tf.concat(values=[i, output_gen1], axis=1) for i in gen1]
    # shape: [CHANNEL_NUM, None, 38, 4, CLASS_NUM // 2, INPUT_LENGTH // 8]
    with tf.device('/cpu:0'):
        output_gen2, gen2 = zip(*[generator2(inputs=gen1[i], encode=encode, \
                                            num=i, train=train) for i in range(CHANNEL_NUM)])
    # shape: [CHANNEL_NUM, None, 4, CLASS_NUM, INPUT_LENGTH // 4]
    # shape: [CHANNEL_NUM, None, 16, 4, CLASS_NUM, INPUT_LENGTH // 4]
    output_gen2 = tf.stack(output_gen2, axis=1, name='output_gen2_stack')
    # shape: [None, CHANNEL_NUM, 4, CLASS_NUM, INPUT_LENGTH // 4]
    gen2 = [tf.concat(values=[i, output_gen2], axis=1) for i in gen2]
    # shape: [CHANNEL_NUM, None, 22, 4, CLASS_NUM, INPUT_LENGTH // 4]
    with tf.device('/cpu:0'):
        output_gen3 = [generator3(inputs=gen2[i], encode=encode, \
                                    num=i, train=train) for i in range(CHANNEL_NUM)]
    # shape: [CHANNEL_NUM, None, CLASS_NUM, INPUT_LENGTH]
    output_gen3 = tf.stack(output_gen3, axis=1, name='output_gen3_stack')
    # shape: [None, CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH]
    print('Generators set')
    with tf.device('/cpu:0'):
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
    tf.add_to_collection(name='loss_dis1', value=loss_dis1)
    tf.add_to_collection(name='loss_dis2', value=loss_dis2)
    tf.add_to_collection(name='loss_dis3', value=loss_dis3)
    tf.add_to_collection(name='loss_gen', value=loss_gen)
    print('Losses set')

def make_iterator():
    filename = 'Dataset/dataset.tfrecord'
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
    def _parse(example_proto):
        feature = {'roll' : tf.FixedLenFeature((6, 72, 384), tf.float32)}
        parsed = tf.parse_single_example(example_proto, feature)
        return parsed['roll']
    dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=16384))
    dataset = dataset.apply(data.map_and_batch(_parse, batch_size=BATCH_SIZE, \
                                        num_parallel_batches=8, drop_remainder=True))
    dataset = dataset.prefetch(16)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def tower_loss(scope, inputs):
    inference(inputs)

    loss_dis1 = tf.add_n(tf.get_collection('loss_dis1', scope=scope), name='sum_loss_dis1')
    loss_dis2 = tf.add_n(tf.get_collection('loss_dis2', scope=scope), name='sum_loss_dis2')
    loss_dis3 = tf.add_n(tf.get_collection('loss_dis3', scope=scope), name='sum_loss_dis3')
    loss_gen = tf.add_n(tf.get_collection('loss_gen', scope=scope), name='sum_loss_gen')
    
    tf.summary.scalar('sum_loss_dis1', loss_dis1)
    tf.summary.scalar('sum_loss_dis2', loss_dis2)
    tf.summary.scalar('sum_loss_dis3', loss_dis3)
    tf.summary.scalar('sum_loss_gen', loss_gen)

    return loss_dis1, loss_dis2, loss_dis3, loss_gen

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
        gen_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.99)

        iterator = make_iterator()

        tower_grads_dis1 = []
        tower_grads_dis2 = []
        tower_grads_dis3 = []
        tower_grads_gen = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        inputs = iterator.get_next()

                        loss_dis1, loss_dis2, loss_dis3, loss_gen = tower_loss(scope, inputs)
                        tf.get_variable_scope().reuse_variables()
                        grad_dis1 = dis1_opt.compute_gradients(loss_dis1)
                        grad_dis2 = dis2_opt.compute_gradients(loss_dis2)
                        grad_dis3 = dis3_opt.compute_gradients(loss_dis3)
                        grad_gen = gen_opt.compute_gradients(loss_gen)

                        tower_grads_dis1.append(grad_dis1)
                        tower_grads_dis1.append(grad_dis2)
                        tower_grads_dis1.append(grad_dis3)
                        tower_grads_dis1.append(grad_gen)

        grads_dis1 = average_gradients(tower_grads_dis1)
        grads_dis2 = average_gradients(tower_grads_dis2)
        grads_dis3 = average_gradients(tower_grads_dis3)
        grads_gen = average_gradients(tower_grads_gen)

        dis1_grad_op = dis1_opt.apply_gradients(grads_dis1)
        dis2_grad_op = dis1_opt.apply_gradients(grads_dis2)
        dis3_grad_op = dis1_opt.apply_gradients(grads_dis3)
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

                for i in range(TRAIN_RATIO_GEN):
                    summary, _, loss_val_gen = sess.run([merged, gen_grad_op, loss_gen], feed_dict=feed_dict, options=run_options)
            
                writer.add_summary(summary, train_count)
                tqdm.write('%06d' % train_count, end=' ')
                tqdm.write('Discriminator1 loss : %.7f' % loss_val_dis1, end=' ')
                tqdm.write('Discriminator2 loss : %.7f' % loss_val_dis2, end=' ')
                tqdm.write('Discriminator3 loss : %.7f' % loss_val_dis3, end=' ')
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