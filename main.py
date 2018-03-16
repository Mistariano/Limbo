# coding:utf-8
__author__ = 'Mistariano'

import tensorflow as tf
from inv_net import inv_net
from vgg import preprocess_tensor, deprocess_tensor, vgg19_and_swap, vgg19
from imageio import imread, imsave, mimread, mimsave, get_reader, get_writer, mvolread
from swap import swap_with_patches, extract_style_patches
import time
import config
import argparse


def transfer_target_media(mode, output_path, model_path=config.MODEL_PATH, style_image_path=config.STYLE_IMAGE_PATH,
                          content_media_path=config.CONTENT_IMAGE_PATH, patch_size=config.PATCH_SIZE,
                          image_size=config.IMAGE_SIZE):
    print('Transfer ({}) with mode: {}'.format(content_media_path, mode))
    s = imread(style_image_path)
    if mode == 'jpg' or mode == 'png':
        mode = 'jpg'
        c = imread(content_media_path)
        height, width, _ = c.shape
    elif mode == 'gif':
        c = mimread(content_media_path)
        height, width, _ = c[0].shape
        print(len(c), 'frames.')
    elif mode == 'mp4':
        print('Reading vedio...')
        c = get_reader(content_media_path)
        fps = c.get_meta_data()['fps']
        shape = c.get_meta_data()['size']
        print(c.get_meta_data())
        print('Done.')
        width, height = shape
    else:
        height, width = 0, 0

    content_img_width = min(image_size, image_size * width // height)
    if mode == 'mp4':
        if content_img_width % 16 != 0:
            content_img_width = content_img_width >> 4 << 4
    content_img_height = content_img_width * height // width
    content_img_size = (content_img_height, content_img_width)
    height, width, _ = s.shape
    style_img_height = min(256, height * 256 // width)
    style_img_width = style_img_height * width // height
    style_img_size = (style_img_height, style_img_width)
    g = tf.Graph()
    with g.as_default():
        style_img_placeholder = tf.placeholder(tf.float32, [None, None, 3])
        content_img_placeholder = tf.placeholder(tf.float32, [None, None, 3])
        processed_style = preprocess_tensor(style_img_placeholder, style_img_size)
        processed_content = preprocess_tensor(content_img_placeholder, content_img_size)

        vgg_out_content = vgg19(processed_content)
        vgg_out_style = vgg19(processed_style)

        style_patches = extract_style_patches(vgg_out_style, patch_size=patch_size)

        style_patches_placeholder = tf.placeholder(tf.float32, style_patches.shape)

        swapped = swap_with_patches(content=vgg_out_content, style_patches=style_patches_placeholder,
                                    patch_size=patch_size, one_hot=False)

        inv_out_features = inv_net(swapped)

        output_imgs = deprocess_tensor(inv_out_features)
        mixed_img = output_imgs[0]

        saver = tf.train.Saver()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = False

        with tf.Session(config=sess_config) as sess:
            saver.restore(sess, model_path)
            sess.run(tf.local_variables_initializer())
            g.finalize()
            print('Using inv-net from', model_path)

            if mode == 'jpg':
                start_time = time.time()
                if c.shape[2] == 4:
                    print('Warning: the image has 4 channels, and the Alpha channel will be ignored.')
                    c = c[:, :, :3]
                patches = sess.run(style_patches, feed_dict={style_img_placeholder: s}, options=run_options)
                out_ = sess.run(mixed_img, feed_dict={style_patches_placeholder: patches, content_img_placeholder: c})
                end_time = time.time()
                print("Takes %.2fs" % (end_time - start_time))
                imsave(output_path, out_)
            elif mode == 'gif':
                c_out = []
                patches = sess.run(style_patches, feed_dict={style_img_placeholder: s}, options=run_options)
                for i in range(len(c)):
                    c_rgb = c[i][:, :, :3]
                    start_time = time.time()
                    out_ = sess.run(mixed_img,
                                    feed_dict={style_patches_placeholder: patches, content_img_placeholder: c_rgb})
                    end_time = time.time()
                    c_out.append(out_)
                    print('Frame', i, "takes %.2fs" % (end_time - start_time))
                mimsave(output_path, c_out, 'GIF', duration=0.1)
            elif mode == 'mp4':
                c_out = []
                writer = get_writer(output_path, fps=fps)
                patches = sess.run(style_patches, feed_dict={style_img_placeholder: s}, options=run_options)
                for i, c_rgb in enumerate(c):
                    start_time = time.time()
                    out_ = sess.run(mixed_img,
                                    feed_dict={style_patches_placeholder: patches, content_img_placeholder: c_rgb})
                    end_time = time.time()
                    c_out.append(out_)
                    print('Frame', i, "takes %.2fs" % (end_time - start_time))
                    writer.append_data(out_)
                    print('Wrote.')
                writer.close()

            print('Done.')
