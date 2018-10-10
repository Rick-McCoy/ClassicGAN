import tensorflow as tf

weight_init = tf.random_normal_initializer()
bias_init = tf.constant_initializer(0)

def pool(input, k=2):
    return tf.layers.max_pooling2d(
        input, pool_size=k, strides=k, data_format='channels_first'
    )

def resize(input, dims=None):
    input = tf.transpose(input, perm=(0, 2, 3, 1))
    if dims is None:
        dims = int(input.get_shape()[1] * 2), int(input.get_shape()[2] * 2)
    input = tf.image.resize_nearest_neighbor(input, dims)
    return tf.transpose(input, perm=(0, 3, 1, 2))

def conv2d(input, out_channels, filter_size=3, k=1, weight_norm=False):
    if len(input.get_shape()) == 3:
        input = tf.expand_dims(input, axis=1)
    filter = tf.get_variable(
        'filter',
        [filter_size, filter_size, input.get_shape()[1], out_channels],
        initializer=weight_init
    )
    if weight_norm:
        mean = tf.reduce_mean(filter)
        c = tf.sqrt(tf.reduce_sum(tf.square(mean - filter)) / (2 * out_channels))
        filter = filter / (c + 1e-7)
    else:
        filter = filter * tf.sqrt(2 / (filter_size ** 2 * int(input.get_shape()[1])))
    b = tf.get_variable('bias', out_channels, initializer=bias_init)
    output = tf.nn.conv2d(input, filter, [1, 1, k, k], padding='SAME', data_format='NCHW')
    output = tf.nn.bias_add(output, b, data_format='NCHW')
    if out_channels == 1:
        output = tf.squeeze(output, axis=1)
    return output

def conv2d_transpose(
        input, 
        output_shape, 
        filter_size=4, 
        k=1, 
        weight_norm=False, 
        w_init=weight_init, 
        b_init=bias_init
    ):
    filter = tf.get_variable(
        'filter', 
        [filter_size // 2, filter_size * 2, output_shape[1], input.get_shape()[1]], 
        initializer=w_init
    )
    if weight_norm:
        mean = tf.reduce_mean(filter)
        c = tf.sqrt(tf.reduce_sum(tf.square(mean - filter)) / (2 * output_shape[1]))
        filter = filter / (c + 1e-7)
    else:
        filter = filter * tf.sqrt(2 / (filter_size ** 2 * int(input.get_shape()[1])))
    b = tf.get_variable('b', output_shape[1], initializer=b_init)
    output = tf.nn.conv2d_transpose(
        input, 
        filter, 
        output_shape, 
        [1, 1, k, k], 
        padding='VALID', 
        data_format='NCHW'
    )
    output = tf.nn.bias_add(output, b, data_format='NCHW')
    return output

def dense_layer(
        input, 
        output_size, 
        weight_norm=False, 
        w_init=weight_init, 
        b_init=bias_init
    ):
    W = tf.get_variable('W', [input.get_shape()[-1], output_size], initializer=w_init)
    if weight_norm:
        mean = tf.reduce_mean(W)
        c = tf.sqrt(tf.reduce_sum(tf.square(mean - W)) / (2 * output_size))
        W = W / (c + 1e-7)
    else:
        W = W * tf.sqrt(2 / int(input.get_shape()[-1]))
    b = tf.get_variable('b', output_size, initializer=b_init)
    return tf.nn.bias_add(tf.matmul(input, W), b)

def leaky_relu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)

def dropout(input, kp=0.75):
    return tf.nn.dropout(input, keep_prob=kp)

def pixelwise_norm(input):
    N = int(input.get_shape()[1])
    sigma = tf.tile(
        tf.reduce_sum(tf.square(input), 1, keepdims=True), 
        [1, tf.shape(input)[1], 1, 1]
    )
    return input / tf.sqrt((1 / N) * sigma + 1e-8)

def minibatch_stddev(input):
    shape = tf.shape(input)
    x_ = tf.tile(tf.reduce_mean(input, 0, keepdims=True), [shape[0], 1, 1, 1])
    sigma = tf.sqrt(tf.reduce_mean(tf.square(input - x_), 0, keepdims=True) + 1e-4)
    sigma_avg = tf.reduce_mean(sigma, keepdims=True)
    layer = tf.tile(sigma_avg, [shape[0], 1, shape[2], shape[3]])
    return tf.concat((input, layer), 1)