import tensorflow as tf

def TemporalBlock(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate ):
    prev_x = x
    init = tf.variance_scaling_initializer(distribution = "uniform")
    padding_size = (kernel_size - 1) * dilation_rate
    # block1
    x = tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]])
    x = tf.layers.conv1d(x, filters=nb_filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init, activation = None)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=dropout_rate)
    # block2
    x = tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]])
    x = tf.layers.conv1d(x, filters=nb_filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init, activation=None)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=dropout_rate)

    if prev_x.shape[-1] != x.shape[-1]:
        prev_x = tf.layers.conv1d(prev_x, filters=nb_filters, kernel_size=1, padding=padding, kernel_initializer=init)

    return tf.nn.relu(prev_x + x)

def TemporalCN(x, output_size, next_skill, num_channels, dropout, kernel_size):
    num_levels = len(num_channels)
    for i in range(num_levels):
        dilation_rate = 2 ** i
        x = TemporalBlock(x, dilation_rate, num_channels[i], kernel_size, padding='valid', dropout_rate=dropout)
    init = tf.contrib.layers.xavier_initializer()
    x = tf.layers.Dense(output_size, kernel_initializer=init, trainable=True)(x)
    x = tf.nn.dropout(x, keep_prob=dropout)
    outputs = x * next_skill
    return outputs