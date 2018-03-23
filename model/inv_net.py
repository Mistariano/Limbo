# coding:utf-8
__author__ = 'Mistariano'

import tensorflow as tf
from tensorflow.contrib import layers


def weight_variable(shape):
    with tf.name_scope('wights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(x, W_shape, b_shape):
    W = weight_variable(W_shape)
    b = bias_variable(b_shape)
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def bilinear_unsampling(x, output_shape):
    """
    :param x: the input tensor
    :param output_shape: [height,width]
    :return: a tensor
    """
    with tf.name_scope('bili-unsampling'):
        # return tf.image.resize_nearest_neighbor(x, size=output_shape)
        return tf.image.resize_bilinear(x, size=output_shape)
        # return tf.image.resize_bicubic(x, size=output_shape)


def conv_instance_norm_relu(x, W_shape, b_shape):
    """
    :param x:
    :param W_shape: [H, W, input, output] ([5, 5, 32, 64])
    :param b_shape: [output] ([64])
    :return:
    """
    with tf.name_scope('conv_instance-norm_relu'):
        with tf.name_scope('conv2d'):
            x = conv2d(x, W_shape, b_shape)
        with tf.name_scope('instance-normalization'):
            x = layers.instance_norm(x)
        with tf.name_scope('relu'):
            x = tf.nn.relu(x)
        return x


class InverseNet:
    def __init__(self, x):
        _, img_H, img_W, _ = x.shape
        img_H *= 4
        img_W *= 4
        with tf.name_scope('inv-net'):
            self.conv_norm_relu_1 = conv_instance_norm_relu(x, [3, 3, 256, 128], [128])
            self.upsampling_1 = bilinear_unsampling(self.conv_norm_relu_1, [img_H // 2, img_W // 2])
            self.conv_norm_relu_2 = conv_instance_norm_relu(self.upsampling_1, [3, 3, 128, 128], [128])
            self.conv_norm_relu_3 = conv_instance_norm_relu(self.conv_norm_relu_2, [3, 3, 128, 64], [64])
            self.upsampling_2 = bilinear_unsampling(self.conv_norm_relu_3, [img_H, img_W])
            self.conv_norm_relu_4 = conv_instance_norm_relu(self.upsampling_2, [3, 3, 64, 64], [64])
            self.out_img = conv2d(self.conv_norm_relu_4, [3, 3, 64, 3], [3])
