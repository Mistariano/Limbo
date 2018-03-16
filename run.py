# coding:utf-8
__author__ = 'Mistariano'

import argparse
import os
from main import transfer_target_media
from config import *
from train import train_inv_net

parser = argparse.ArgumentParser()
parser.add_argument('-M', '--mode', help='', default='auto', choices=['auto', 'jpg', 'gif', 'mp4'])
parser.add_argument('--train', default=False, action='store_true')
args = parser.parse_args()
print(args)
if not args.train:
    for content_filename in os.listdir(CONTENT_DIR):
        content_path = os.path.join(CONTENT_DIR, content_filename)
        output_path = os.path.join(OUTPUT_DIR, content_filename)
        suffix = os.path.splitext(content_filename)[1][1:].lower()
        transfer_target_media(mode=suffix, content_media_path=content_path, output_path=output_path)
else:
    print('train a new inverse net? [y/n]')
    s = input()
    if s == 'y':
        train_inv_net()
    else:
        print('Cancelled.')
