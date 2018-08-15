import tensorflow as tf

n_classes = 10
image_size = 32

dropout = tf.placeholder(tf.float32, name="dropout_rate")
input_images = tf.placeholder(tf.float32,
                              shape=[None, image_size, image_size, 3],
                              name="input_images")

# Network Size
first_conv_size = 96
second_conv_size = 256
third_conv_size = 384
fourth_conv_size = 384
fifth_conv_size = 256

# First CONV layer
kernel = tf.Variable(tf.truncated_normal([11, 11, 3, first_conv_size],
                                         dtype=tf.float32,
                                         stddev=1e-1),
                     name="conv1_weights")
conv = tf.nn.conv2d(input_images, kernel, [1, 4, 4, 1], padding="SAME")
bias = tf.Variable(tf.truncated_normal([first_conv_size]))
conv_with_bias = tf.nn.bias_add(conv, bias)
conv1 = tf.nn.relu(conv_with_bias, name="conv1")

lrn1 = tf.nn.lrn(conv1,
                 alpha=1e-4,
                 beta=0.75,
                 depth_radius=2,
                 bias=2.0)

pooled_conv1 = tf.nn.max_pool(lrn1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pool1")

# Second CONV Layer
kernel = tf.Variable(tf.truncated_normal([5, 5, first_conv_size, second_conv_size],
                                         dtype=tf.float32,
                                         stddev=1e-1),
                     name="conv2_weights")
conv = tf.nn.conv2d(pooled_conv1, kernel, [1, 4, 4, 1], padding="SAME")
bias = tf.Variable(tf.truncated_normal([second_conv_size]), name="conv2_bias")
conv_with_bias = tf.nn.bias_add(conv, bias)
conv2 = tf.nn.relu(conv_with_bias, name="conv2")
lrn2 = tf.nn.lrn(conv2,
                 alpha=1e-4,
                 beta=0.75,
                 depth_radius=2,
                 bias=2.0)

pooled_conv2 = tf.nn.max_pool(lrn2,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME",
                              name="pool2")

# Third CONV layer
kernel = tf.Variable(tf.truncated_normal([3, 3, second_conv_size, third_conv_size],
                                         dtype=tf.float32,
                                         stddev=1e-1),
                     name="conv3_weights")
conv = tf.nn.conv2d(pooled_conv2, kernel, [1, 1, 1, 1], padding="SAME")
bias = tf.Variable(tf.truncated_normal([third_conv_size]), name="conv3_bias")
conv_with_bias = tf.nn.bias_add(conv, bias)
conv3 = tf.nn.relu(conv_with_bias, name="conv3")

# Fourth CONV layer
kernel = tf.Variable(tf.truncated_normal([3, 3, third_conv_size, fourth_conv_size],
                                         dtype=tf.float32,
                                         stddev=1e-1),
                     name="conv4_weights")
conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
bias = tf.Variable(tf.truncated_normal([fourth_conv_size]), name="conv4_bias")
conv_with_bias = tf.nn.bias_add(conv, bias)
conv4 = tf.nn.relu(conv_with_bias, name="conv4")

# Fifth CONV Layer
kernel = tf.Variable(tf.truncated_normal([3, 3, fourth_conv_size, fifth_conv_size],
                                         dtype=tf.float32,
                                         stddev=1e-1),
                     name="conv5_weights")
conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
bias = tf.Variable(tf.truncated_normal([fifth_conv_size]), name="conv5_bias")
conv_with_bias = tf.nn.bias_add(conv, bias)
conv5 = tf.nn.relu(conv_with_bias, name="conv5")

# Fully Connected Layers
fc_size = fifth_conv_size
conv5 = tf.layers.flatten(conv5) # tf.flatten
weights = tf.Variable(tf.truncated_normal([fc_size, fc_size]), name="fc1_weights")
bias = tf.Variable(tf.truncated_normal([fc_size]), name="fc1_bias")
fc1 = tf.matmul(conv5, weights) + bias
fc1 = tf.nn.relu(fc1, name="fc1")
fc1 = tf.nn.dropout(fc1, dropout)

weights = tf.Variable(tf.truncated_normal([fc_size, fc_size]), name="fc2_weights")
bias = tf.Variable(tf.truncated_normal([fc_size]), name="fc2_bias")
fc2 = tf.matmul(fc1, weights) + bias
fc2 = tf.nn.relu(fc2, name="fc2")
fc2 = tf.nn.dropout(fc2, dropout)

weights = tf.Variable(tf.zeros([fc_size, n_classes]), name="output_weight")
bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
out = tf.matmul(fc2, weights) + bias
