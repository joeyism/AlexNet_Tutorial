from cifar import Cifar
import tensorflow as tf
import model
import helper

learning_rate = 0.001
batch_size = 16
no_of_epochs = 10

y = tf.placeholder(tf.float32, [None, model.n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model.out,
    labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cifar = Cifar(batch_size=batch_size)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    acc = 0
    loss = 0
    for epoch in range(no_of_epochs):
            for batch in tqdm(cifar.batches,
                          desc="Epoch {}".format(epoch),
                          unit="batch"):

        inp, out = helper.transform_to_input_output(batch, dim=model.n_classes)

        merge = tf.summary.merge_all()
        summary, _ = sess.run([merge, optimizer],
                    feed_dict={
                        model.input_images: inp,
                        y: out})

        acc = sess.run(accuracy,
                       feed_dict={
                           model.input_images: inp,
                           y: out})

        loss = sess.run(cost,
                        feed_dict={
                            model.input_images: inp,
                            y: out})

        print("Acc: {} Loss: {}".format(acc, loss))
