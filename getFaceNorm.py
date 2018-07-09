# --coding utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import os.path as osp

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import struct

import numpy as np
import cv2
from easydict import EasyDict as edict

import mxnet as mx


if cv2.__version__.startswith('3.'):
    IMREAD_AS_GRAY = cv2.IMREAD_GRAYSCALE
    IMREAD_AS_COLOR = cv2.IMREAD_COLOR
else:
    IMREAD_AS_GRAY = cv2.CV_LOAD_IMAGE_GRAYSCALE
    IMREAD_AS_COLOR = cv2.CV_LOAD_IMAGE_COLOR


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, IMREAD_AS_GRAY)
    else:
        img = cv2.imread(img_path, IMREAD_AS_COLOR)
        if mode == 'rgb':
            #print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))

    print('img.shape: ', img.shape)
    return img


def init_input_blob(batch_size, image_shape):
    input_shape = (batch_size, image_shape[0], image_shape[1], image_shape[2])
    input_blob = np.zeros(input_shape, dtype=np.float32)

    print('init_input_blob->id(input_blob): ', id(input_blob))

    return input_blob


def load_image_data(image_path, input_blob, idx, image_shape, use_mean=True):
    print('load_image_data->id(input_blob): ', id(input_blob))

    img = read_image(image_path, mode='rgb')
    # print(img.shape)
    if img is None:
        print('parse image', image_path, 'error')
        return None

    if img.shape != (image_shape[1], image_shape[2], image_shape[0]):
        print('image shape must be: ',
              (image_shape[1], image_shape[2], image_shape[0]))
        return None

    if use_mean > 0:
        print('remove mean and normalize into [-1, 1]')

        v_mean = np.array([127.5, 127.5, 127.5],
                          dtype=np.float32).reshape((1, 1, 3))
        img = img.astype(np.float32) - v_mean
        img *= 0.0078125

    input_blob[idx] = np.transpose(img, (2, 0, 1))

    return True


def add_flip_to_input_blob(input_blob):
    batch_size_half = int(input_blob.shape[0] / 2)
    n_channels = input_blob.shape[1]
    # print('===> In add_flip_to_input_blob():')
    # print('batch_size_half: ', batch_size_half)
    for i in range(batch_size_half):
        # print('---> before flip:')

        # for j in range(n_channels):
        #     print('---> input_blob[{},{},0:3,0:3] = {}'.format(
        #             i, j, input_blob[i, j, 0:3, 0:3]))
        # print('\n')

        # add flipped data
        for j in range(n_channels):
            input_blob[i +
                       batch_size_half][j][...] = np.fliplr(input_blob[i][j][...])

        # do_flip(input_blob[i + batch_size_half])

        # print('---> after flip:')
        # for j in range(n_channels):
        #     print('---> input_blob[batch_size_half+{},{},0:3,0:3] = {}'.format(
        #         i, j, input_blob[batch_size_half + i, j, 0:3, 0:3]))
        #     print('---> input_blob[batch_size_half+{},{},0:3,-3:] = {}'.format(
        #         i, j, input_blob[batch_size_half + i, j, 0:3, -3:]))
        # print('\n')

    # for i in range(batch_size_half):
    #     for j in range(n_channels):
    #         print('---> input_blob[{},{},0:3,0:3] = {}'.format(
    #             i, j, input_blob[i, j, 0:3, 0:3]))
    #         print('     input_blob[batch_size_half+{},{},0:3,-3] = {}'.format(
    #             i, j, input_blob[batch_size_half + i, j, 0:3, -3:]))


def get_predict_label(input_blob, net, n_imgs=None):
    if not n_imgs:
        n_imgs = input_blob.shape[0]

    print('get_features->id(input_blob): ', id(input_blob))

    batch_size_half = int(input_blob.shape[0] / 2)

    print('n_imgs: ', n_imgs)

    data = mx.nd.array(input_blob)
    print('type(data): ', type(data))
    print('data.shape: ', data.shape)

    # for i in range(n_imgs):
    #     for j in range(3):
    #         print('---> data[{},{},0:3,0:3] = {}'.format(
    #             i, j, data[i, j, 0:3, 0:3]))
    #         print('     data[batch_size_half+{},{},0:3,-3] = {}'.format(
    #             i, j, data[batch_size_half + i, j, 0:3, -3:]))

    db = mx.io.DataBatch(data=(data,))

    net.model.forward(db, is_train=False)
    #outputs = net.model.get_outputs()[0]
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    outputs = net.model.get_outputs()[0]
    print('outputs.shape: ', outputs.shape)
    result = []
    for j in range(n_imgs):
        embedding = outputs[j].asnumpy().flatten()
        norm = np.linalg.norm(embedding)
        result.append(norm)
    return result


def main(args):
    print('===> args:\n', args)

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fail_log_fn = osp.join(save_dir, 'failed_image_list.txt')
    fail_fp = open(fail_log_fn, 'w')

    gpuid = args.gpu
    ctx = mx.gpu(gpuid)

    image_shape = [int(x) for x in args.image_size.split(',')]

    batch_size = args.batch_size

    if batch_size < 1:
        batch_size = 1

    input_batch_size = batch_size
    model = args.model

    vec = model.split(',')
    assert len(vec) > 1
    prefix = vec[0]
    epoch = int(vec[1])  # type: int
    print('loading', prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
    net.model.bind(data_shapes=[('data', (input_batch_size, 3, image_shape[1], image_shape[2]))])
    net.model.set_params(net.arg_params, net.aux_params)

    img_list_fn = args.image_list
    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    img_list = []

    input_blob = init_input_blob(input_batch_size, image_shape)

    i = 0
    load_idx = 0
    batch_cnt = 0

    pre_label_file = open(osp.join(save_dir, 'pre_image_list.txt'), 'w')

    for line in open(img_list_fn, 'r'):
        if i % 1000 == 0:
            print("\n===> Try to process %d images, %d succeeded"
                  % (i, batch_cnt * batch_size + load_idx))

        i += 1

        image_path = line.strip()
        full_path = osp.join(args.image_dir, image_path)
        print('\n---> Loading ', full_path)

        ret = load_image_data(full_path, input_blob,
                              load_idx, image_shape, args.use_mean)
        load_idx += 1

        if not ret:
            print('---> Failed to load: ', full_path)
            fail_fp.write(image_path + '\n')
            continue

        img_list.append(image_path)

        if load_idx == batch_size:
            fail_fp.flush()

            batch_cnt += 1
            print('\n---> Get features for batch #', batch_cnt)

            pre_labels = get_predict_label(input_blob, net, len(img_list))

            for j, fn in enumerate(img_list):
                pre_label_file.write(str(fn) + ' ' + str(pre_labels[j]) + '\n')

            img_list = []
            load_idx = 0

    if len(img_list) > 0:
        print("\n===> Try to process %d images, %d succeeded"
              % (i, batch_cnt * batch_size + load_idx))

        batch_cnt += 1
        print('\n---> Get features for batch #', batch_cnt)

        pre_labels = get_predict_label(input_blob, net, len(img_list))

        for j, fn in enumerate(img_list):
            pre_label_file.write(str(fn) + ' ' + str(pre_labels[j]) + '\n')

    print("===> Try to process %d images, %d succeeded"
          % (i, batch_cnt * batch_size + load_idx))
    fail_fp.close()
    pre_label_file.close()


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--image-dir', type=str,
                        help='image root dir if image list contains relative paths')
    parser.add_argument('--save-dir', type=str,
                        help='save path of predict label ')
    parser.add_argument('--batch-size', type=int, help='', default=1)
    parser.add_argument('--image-size', type=str,
                        help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--use-mean', action='store_true')
    parser.add_argument('--model', type=str, help='',
                        default='../model/model-r50-am-lfw/model,0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
