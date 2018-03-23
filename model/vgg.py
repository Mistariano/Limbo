# coding:utf-8

import tensorflow as tf
import scipy.io as sio
import numpy as np
import PIL.Image as Image
import scipy
from model.swap import swap_batch
from config import VGG_PATH
import os
import sys

data_path = VGG_PATH
if not os.path.exists(data_path):
    print('Error: VGG-19 is not loaded because there\'s no file at {}'.format(data_path))
    sys.exit(1)

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

mean_pixel = np.array([123.68, 116.779, 103.939], np.float32)
_vgg_params = None


def load_net(data_path):
    global _vgg_params
    if _vgg_params is None:
        _vgg_params = scipy.io.loadmat(data_path)
    # mean = data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))
    weights = _vgg_params['layers'][0]
    return weights, mean_pixel


def net_preloaded(weights, input_image, pooling='avg'):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current
    return net


def net_preloaded_with_swap(weights, input_image, patch_size_dict, pooling='avg', use_one_hot=False, with_style=True):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        if name in patch_size_dict.keys():
            current = swap(current, patch_size=patch_size_dict[name], one_hot=use_one_hot, with_style=with_style)
        net[name] = current
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, tf.constant(bias))


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME')


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = np.cast(image, np.float32)
    # image = image - mean_pixel
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_tensor(image_tensor, size):
    image_tensor = tf.cast(image_tensor, tf.float32)
    # image_tensor = image_tensor - tf.constant(mean_pixel)
    resized = tf.expand_dims(image_tensor, 0)
    return tf.image.resize_images(resized, size)


def deprocess_tensor(image_tensor):
    # image_tensor = image_tensor + tf.constant(mean_pixel)
    # image_tensor += tf.constant([255.0, 255.0, 255.0])
    cliped = tf.clip_by_value(image_tensor, 0, 255)
    casted = tf.cast(cliped, tf.uint8)
    # features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # gram = K.dot(features, K.transpose(features))
    # return gram
    return casted


def deprocess_single_image(image):
    shape = image.shape
    image = image.reshape(shape[1:])
    # image += mean_pixel
    # image += [255.0, 255.0, 255.0]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def vgg19(input_img, output_layer='conv3_1'):
    if output_layer not in VGG19_LAYERS:
        print('Error, there\'s no layer named', output_layer)
    weights, mean = load_net(data_path)
    return net_preloaded(weights, input_img)[output_layer]


def vgg19_and_swap(input_img, output_layer='conv3_1', patch_size_dict=None, use_one_hot=False, with_style=True):
    if output_layer not in VGG19_LAYERS:
        print('Error, there\'s no layer named', output_layer)
    weights, mean = load_net(data_path)
    if patch_size_dict is not None:
        for name, arg in patch_size_dict.items():
            print('Swap style with %d^2 patch' % arg)
    else:
        patch_size_dict = {}
    if len(patch_size_dict.keys()) > 1 and not with_style:
        print('Warn: Must keep style feature if more than 1 swap will be done.')
        with_style = True
    return net_preloaded_with_swap(weights, input_img, patch_size_dict=patch_size_dict, use_one_hot=use_one_hot,
                                   with_style=with_style)[output_layer]
