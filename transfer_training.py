from cifar import Cifar
from tqdm import tqdm
import tensorflow as tf
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

tf.summary.histogram("cost", cost)
tf.summary.histogram("accuracy", accuracy)

cifar = Cifar(batch_size=batch_size)

init = tf.initialize_all_variables()

saver = tf.train.Saver()
i = 0
with tf.Session() as sess:
    sess.run(init)

    reader = tf.WholeFileReader()
    writer = tf.summary.FileWriter( './logs/', sess.graph)

    for epoch in range(no_of_epochs):
        for batch in tqdm(cifar.batches,
                          desc="Epoch {}".format(epoch),
                          unit="batch"):

            inp, out = helper.transform_to_input_output_and_pad(batch, dim=n_classes)

            sess.run([optimizer],
                        feed_dict={
                            pretrained.x: inp,
                            y: out,
                            })

        merge = tf.summary.merge_all()
        acc, loss, summary = sess.run([accuracy, cost, merge],
                       feed_dict={
                           pretrained.x: inp,
                           y: out,
                           })

        writer.add_summary(summary, i)
        i = i + 1

        print("Acc: {} Loss: {}".format(acc, loss))

        inp_test, out_test = helper.transform_to_input_output_and_pad(cifar.test_set,
                                                        dim=n_classes)

        test_acc = sess.run([accuracy],
                feed_dict={
                    pretrained.x: inp_test,
                    y: out_test,
                    })
        print("Test Acc: {}".format(test_acc))

        saver.save(sess, "saved_model/alexnet.ckpt")

