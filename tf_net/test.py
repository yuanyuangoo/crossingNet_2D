import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def binary_activation(x, r):
    cond = tf.less(x, r*tf.ones(tf.shape(x)))
    out = tf.where(cond, -1*tf.ones(tf.shape(x)), tf.ones(tf.shape(x)))
    return out

    
tf.image.random_brightness


x = tf.constant((0.8, 0.4,111), dtype=tf.float32)
a = binary_activation(x, 0.5)
with tf.Session() as sess:
    print(a.eval())
