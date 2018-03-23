# coding:utf-8
__author__ = 'Mistariano'

VGG_PATH = 'model/data/imagenet-vgg-verydeep-19.mat'

STYLE_IMAGE_PATH = 'style.jpg'
CONTENT_IMAGE_PATH = 'base.jpg'
RUN_INVERSE_NET_PATH = 'model/data/inverse_net_patch1/inv_net.ckpt'
IMAGE_SIZE = 400
PATCH_SIZE = 3

STYLE_DIR = 'styles'
CONTENT_DIR = 'contents'
OUTPUT_DIR = 'results'

# Train default config
TRAIN_NATURE_IMAGE_PATH = 'D://projects/data/MSCOCO/train2014/'
TRAIN_ART_IMAGE_PATH = 'D://projects/data/Painter by numbers/train/'
TRAIN_INVERSE_NET_SAVE_PATH = 'model/inverse_net_patch1/inv_net.ckpt'
TRAIN_LOG_DIR = 'log1'
TRAIN_IMAGE_SIZE = (256, 256)
TRAIN_PARAMS_EPOCHS = 2
TRAIN_PARAMS_ITERATION = 80000
TRAIN_PARAMS_LAMBDA_TV = 1e-6
