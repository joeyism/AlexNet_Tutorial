from cifar import Cifar
from tqdm import tqdm
import tensornets as nets
import tensorflow as tf
import numpy as np
import helper

learning_rate = 0.00001
batch_size = 16
no_of_epochs = 100
n_classes = 10
no_of_test_splits = 100
image_size = 224

inputs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
outputs = tf.placeholder(tf.float32, [None, n_classes])

vgg = nets.VGG19(inputs, is_training=True, classes=n_classes)
model = tf.identity(vgg, name='logits')
cost = tf.losses.softmax_cross_entropy(outputs, vgg)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
vgg.print_outputs()
vgg.print_summary()


cifar = Cifar(batch_size=batch_size)
cifar.create_resized_test_set(dim=n_classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(vgg.pretrained())
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    for epoch in range(no_of_epochs):
        for i in tqdm(range(cifar.no_of_batches),
                desc="Epoch {}".format(epoch),
                unit=" batch "):
            this_batch = cifar.batch(i)
            input_batch, out = helper.reshape_batch(this_batch, (image_size, image_size), n_classes)
            sess.run([optimizer],
                        feed_dict={
                            inputs: input_batch,
                            outputs: out },
                        options=run_options)

        acc, loss = sess.run([accuracy, cost],
                       feed_dict={
                            inputs: input_batch,
                            outputs: out },
                       options=run_options)

        print("Last Batch Acc: {} Loss: {}".format(acc, loss))

        inp_test, out_test = cifar.test_set
        inp_test = np.split(inp_test, no_of_test_splits)
        out_test = np.split(out_test, no_of_test_splits)

        total_acc = 0
        for each_inp_test, each_out_test in tqdm(zip(inp_test, out_test),
                desc="Test".format(epoch),
                unit=" batch ",
                total=no_of_test_splits):

            each_test_acc = sess.run(accuracy,
                    feed_dict={
                        inputs: each_inp_test,
                        outputs: each_out_test},
                    options=run_options)
            total_acc = total_acc + each_test_acc

        test_acc = total_acc / no_of_test_splits
        print("Test Acc: {}".format(test_acc))
