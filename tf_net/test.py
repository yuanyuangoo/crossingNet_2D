import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


input = np.asarray([1, 0, 2])
a = tf.placeholder(dtype=tf.float32, shape=(3))
b = tf.placeholder(dtype=tf.float32, shape=(3))

# b = tf.placeholder(dtype=tf.float32)
with tf.variable_scope("father") as scope:
    with tf.variable_scope("test") as scope:
        c = tf.get_variable("c", shape=(3))
        d = tf.tensordot(a, c, 1)

# with tf.variable_scope("father") as scope:
with tf.variable_scope("father/test") as scope:
    scope.reuse_variables()
    c = tf.get_variable("c", shape=(3))
    f = tf.tensordot(b, c, 1)

loss = tf.square(f-d)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

show_all_variables()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(30):
        # af, cf, df = sess.run([a, c, d], feed_dict={a: input, b: input+1})
        # print(af, cf, df)

        # af, cf, ff = sess.run([a, c, f], feed_dict={a: input, b: input+1})
        # print(af, cf, ff)

        af, bf, cf, df, ff, loss1, _ = sess.run(
            [a, b, c, d, f, loss, optimizer], feed_dict={a: input, b: input+1})

        # print(af, bf, cf, df, ff, loss1)
