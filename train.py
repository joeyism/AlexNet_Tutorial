from cifar import Cifar
import tensorflow as tf
import model

learning_rate = 0.001

y = tf.placeholder(tf.float32, [None, model.n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model.out,
    labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))