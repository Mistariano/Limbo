# coding:utf-8
__author__ = 'Mistariano'

import tensorflow as tf
import PIL.Image as Image


def swap_1(swap_input, with_style=False):
    channel = swap_input.shape[3]
    style_feature = swap_input[1]
    content_image_feature = swap_input[0]
    shape = content_image_feature.shape
    content_image_feature = tf.reshape(content_image_feature, (-1, channel))
    style_feature = tf.reshape(style_feature, (-1, channel))
    style_normed = tf.nn.l2_normalize(style_feature, axis=-1)
    mat = tf.matmul(content_image_feature, style_normed, transpose_b=True, )

    max = tf.reshape(tf.reduce_max(mat, axis=-1), (-1, 1))
    mat = mat - max
    mat = tf.clip_by_value(mat, -50, 50)
    ex = tf.exp(mat)
    ex = tf.check_numerics(ex, 'inf/nan error:exp')
    c = tf.reduce_sum(ex, axis=1)
    prob = ex / tf.reshape(c, (-1, 1))
    prob = tf.check_numerics(prob, 'inf/nan error:divide')
    out = tf.matmul(prob, style_feature)
    out = tf.expand_dims(tf.reshape(out, shape), 0)
    if with_style:
        out = tf.concat((out, tf.expand_dims(tf.reshape(style_feature, shape), 0)), 0)
    return out


def extract_style_patches(style, patch_size=3, channel=256):
    style_patches = tf.extract_image_patches(style, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             rates=[1, 1, 1, 1])
    style_patches = tf.reshape(style_patches, [-1, patch_size, patch_size, channel])
    style_patches = tf.transpose(style_patches, [1, 2, 3, 0])
    return style_patches


def swap_with_patches(content, style_patches, one_hot=False, patch_size=3):
    style_patches_normed = tf.nn.l2_normalize(style_patches, axis=(0, 1, 2))
    conv = tf.nn.conv2d(input=content, filter=style_patches_normed, strides=[1, 1, 1, 1], padding='SAME')
    if one_hot:
        with tf.name_scope('one-hot'):
            argmax = tf.argmax(conv, -1)
            one_hot = tf.one_hot(argmax, depth=conv.shape[-1])
    else:
        with tf.name_scope('softmax'):
            max = tf.reduce_max(conv, axis=-1)
            max = tf.expand_dims(max, -1)
            w = conv - max
            w = tf.clip_by_value(w, -50, 0)
            prob = tf.exp(w)
            sum = tf.expand_dims(tf.reduce_sum(prob, axis=-1), -1)
            one_hot = prob / sum
    decov = tf.nn.conv2d_transpose(one_hot, filter=style_patches, strides=[1, 1, 1, 1],
                                   output_shape=content.shape)
    decov = decov / (patch_size ** 2)
    return decov


def swap(swap_input, patch_size=3, with_style=False, one_hot=False):
    """

    :param swap_input: 0:content, 1:style
    :param img_H:
    :param img_W:
    :param img_H_s:
    :param img_W_s:
    :return:
    """
    if patch_size == 0:
        return swap_input
    elif patch_size == 1:
        return swap_1(swap_input, with_style=with_style)

    channel = swap_input.shape[3]
    style_feature = tf.expand_dims(swap_input[1], axis=0)
    content_image_feature = tf.expand_dims(swap_input[0], axis=0)
    style_patches = tf.extract_image_patches(style_feature, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             rates=[1, 1, 1, 1])
    style_patches = tf.reshape(style_patches, [-1, patch_size, patch_size, channel])
    style_patches = tf.transpose(style_patches, [1, 2, 3, 0])
    style_patches_normed = tf.nn.l2_normalize(style_patches, axis=(0, 1, 2))

    conv = tf.nn.conv2d(input=content_image_feature, filter=style_patches_normed, strides=[1, 1, 1, 1], padding='SAME')
    if one_hot:
        with tf.name_scope('one-hot'):
            argmax = tf.argmax(conv, -1)

            one_hot = tf.one_hot(argmax, depth=conv.shape[-1])
    else:
        with tf.name_scope('softmax'):
            max = tf.reshape(tf.reduce_max(conv, axis=-1), (conv.shape[0], conv.shape[1], conv.shape[2], 1))
            w = conv - max
            w = tf.clip_by_value(w, -50, 0)
            prob = tf.exp(w)
            sum = tf.reshape(tf.reduce_sum(prob, axis=-1), (conv.shape[0], conv.shape[1], conv.shape[2], 1))
            one_hot = prob / sum
            # one_hot=tf.nn.softmax(conv,axis=-1)
    decov = tf.nn.conv2d_transpose(one_hot, filter=style_patches, strides=[1, 1, 1, 1],
                                   output_shape=content_image_feature.shape)
    decov = decov / (patch_size ** 2)
    if with_style:
        decov = tf.concat((decov, style_feature), 0)
    return decov
