# coding:utf-8
__author__ = 'Mistariano'

import tensorflow as tf
from vgg import vgg19, preprocess_image, deprocess_single_image, preprocess_tensor, deprocess_tensor
from swap import swap
from inv_net import inv_net, InverseNet
import time
import os


def train_inv_net():
    nature_img_path = 'D://projects/data/MSCOCO/train2014/'
    style_img_path = 'D://projects/data/Painter by numbers/train/'
    model_save_path = 'model/inverse_net_patch1/inv_net.ckpt'
    log_dir = 'log1'
    img_h = 256
    img_w = 256

    with tf.name_scope('hyper_parameters'):
        num_epochs = 2
        lambda_tv = 1e-6

    with tf.name_scope('read'):
        nature_filenames = tf.train.string_input_producer(
            list(nature_img_path + name for name in os.listdir(nature_img_path)),
            num_epochs=num_epochs, shuffle=True)
        style_filenames = tf.train.string_input_producer(
            list(style_img_path + name for name in os.listdir(style_img_path)),
            num_epochs=num_epochs, shuffle=True)
        reader = tf.WholeFileReader()
        _, nature_img_1_name = reader.read(nature_filenames)
        _, nature_img_2_name = reader.read(nature_filenames)
        _, style_img_1_name = reader.read(style_filenames)
        _, style_img_2_name = reader.read(style_filenames)

        nature_img_1 = tf.image.decode_jpeg(nature_img_1_name, channels=3, )
        nature_img_2 = tf.image.decode_jpeg(nature_img_2_name, channels=3)
        style_img_1 = tf.image.decode_jpeg(style_img_1_name, channels=3)
        style_img_2 = tf.image.decode_jpeg(style_img_2_name, channels=3)

        nature_img_1 = preprocess_tensor(nature_img_1, (img_h, img_w))
        nature_img_2 = preprocess_tensor(nature_img_2, (img_h, img_w))
        style_img_1 = preprocess_tensor(style_img_1, (img_h, img_w))
        style_img_2 = preprocess_tensor(style_img_2, (img_h, img_w))

        input_images = tf.concat((nature_img_1, nature_img_2, style_img_1, style_img_2), 0)

    with tf.name_scope('vgg_features'):
        features = vgg19(input_images)

    with tf.name_scope('swap'):
        f1 = tf.expand_dims(features[0], 0)
        f2 = tf.expand_dims(features[1], 0)
        f3 = tf.expand_dims(features[2], 0)
        f4 = tf.expand_dims(features[3], 0)
        cat1 = tf.concat((f1, f3), 0)
        cat2 = tf.concat((f1, f4), 0)
        cat3 = tf.concat((f2, f3), 0)
        cat4 = tf.concat((f2, f4), 0)
        swapped1 = swap(cat1, patch_size=1)
        swapped2 = swap(cat2, patch_size=1)
        swapped3 = swap(cat3, patch_size=1)
        swapped4 = swap(cat4, patch_size=1)
        features = tf.concat((features, swapped1, swapped2, swapped3, swapped4), 0)

    inverse_net = InverseNet(features, img_h, img_w)
    inv_img = inverse_net.out_img
    image_summary = tf.summary.image('out', deprocess_tensor(inv_img), max_outputs=8, family='train')

    with tf.name_scope('inv_features'):
        inv_features = vgg19(inv_img)

    with tf.name_scope('total'):
        a = inv_img[:, 1:, :, :] - inv_img[:, :img_h - 1, :, :]
        b = inv_img[:, :, 1:, :] - inv_img[:, :, :img_w - 1, :]
        a = tf.pow(a, 2)
        b = tf.pow(b, 2)
        sum1 = tf.reduce_sum(a)
        sum2 = tf.reduce_sum(b)
        loss_tv = sum1 + sum2

    with tf.name_scope('loss'):
        loss = tf.pow(inv_features - features, 2)
        loss = tf.reduce_sum(loss)
        loss = loss + loss_tv * lambda_tv
        loss_summary = tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer().minimize(loss)

    merged = tf.summary.merge([image_summary, loss_summary])
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists(model_save_path + '.meta'):
            saver.restore(sess, model_save_path)
            sess.run(tf.local_variables_initializer())
            print('Restored from', model_save_path)
        else:
            sess.run(init_op)
            print('Init done.')
        tf.train.start_queue_runners(sess=sess)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        for i in range(80000):
            try:
                if i % 1000:
                    start_time = time.time()
                    summary, loss_, _ = sess.run([loss_summary, loss, train_step])
                    end_time = time.time()
                else:
                    start_time = time.time()
                    summary, loss_, _ = sess.run([merged, loss, train_step])
                    end_time = time.time()
            except:
                print(i, 'Error, continue')
            else:
                print(i, loss_, end_time - start_time)
                writer.add_summary(summary, i)
            if i % 1000 == 0:
                saver.save(sess, model_save_path)
                print('Saved at', model_save_path)
        writer.close()
