from cifar import Cifar
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pretrained
import helper


n_classes = 10
learning_rate = 0.001
batch_size = 16
no_of_epochs = 100


conv5 = tf.layers.flatten(pretrained.maxpool5) # tf.flatten


weights = tf.Variable(tf.zeros([9216, n_classes]), name="output_weight")
bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
out = tf.matmul(conv5, weights) + bias

y = tf.placeholder(tf.float32, [None, n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=out,
    labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cifar = Cifar(batch_size=batch_size)
cifar.create_padded_batches(new_size=(227, 227), dim=10)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    for epoch in range(no_of_epochs):
        for batch, out in tqdm(cifar.padded_batches,
                desc="Epoch {}".format(epoch),
                unit=" batch "):

            sess.run([optimizer],
                        feed_dict={
                            pretrained.x: batch,
                            y: out },
                        options=run_options)

        acc, loss = sess.run([accuracy, cost],
                       feed_dict={
                           pretrained.x: batch,
                           y: out },
                       options=run_options)

        print("Acc: {} Loss: {}".format(acc, loss))

        no_of_test_splits = 50
        inp_test, out_test = cifar.padded_test_set
        inp_test = np.split(inp_test, no_of_test_splits)
        out_test = np.split(out_test, no_of_test_splits)

        total_acc = 0
        for each_inp_test, each_out_test in tqdm(zip(inp_test, out_test),
                desc="Test".format(epoch),
                unit=" batch ",
                total=no_of_test_splits):

            each_test_acc = sess.run(accuracy,
                    feed_dict={
                        pretrained.x: each_inp_test,
                        y: each_out_test },
                    options=run_options)
            total_acc = total_acc + each_test_acc

        test_acc = total_acc / no_of_test_splits
        print("Test Acc: {}".format(test_acc))
