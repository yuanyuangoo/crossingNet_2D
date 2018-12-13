
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image = cv2.imread('plain_cup.png', 0)

tf.reset_default_graph()
# Write the kernel weights as a 2D array.
kernel_h = np.array([3, 3])
kernel_h = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
kernel_v = np.array([3, 3])
kernel_v = [[1, 0, 1], [2, 0, -2], [-1, 0, -1]]
# Kernel weights
if len(kernel_h) == 0 or len(kernel_v) == 0:
    print('Please specify the kernel!')
input_placeholder = tf.placeholder(
    dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))


with tf.name_scope('convolution'):
    conv_w_h = tf.constant(kernel_h, dtype=tf.float32, shape=(3, 3, 1, 1))
    conv_w_v = tf.constant(kernel_v, dtype=tf.float32, shape=(3, 3, 1, 1))
    output_h = tf.nn.conv2d(input=input_placeholder, filter=conv_w_h, strides=[
                            1, 1, 1, 1], padding='SAME')
    output_v = tf.nn.conv2d(input=input_placeholder, filter=conv_w_v, strides=[
                            1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    result_h = sess.run(output_h, feed_dict={
        input_placeholder: image[np.newaxis, :, :, np.newaxis]})
    result_v = sess.run(output_v, feed_dict={
        input_placeholder: image[np.newaxis, :, :, np.newaxis]})
    cv2.imwrite("result_h.jpg", result_h[0, :, :, 0])  # view horisontaal edges
    cv2.imwrite("result_v.jpg", result_v[0, :, :, 0])  # view vertical edges


result_lenght = ((result_v**2) + (result_h**2))**0.5
cv2.imwrite("result.jpg", result_lenght[0, :, :, 0])
