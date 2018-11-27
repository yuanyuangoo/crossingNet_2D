import tensorflow as tf
x = 1.0
y = 1.0
with tf.Session() as sess:
    r = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    print(r)
