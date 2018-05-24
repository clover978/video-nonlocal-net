# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import argparse
import sys
import pickle
import datetime
import os
import math
import cv2
from sets import Set
from collections import defaultdict

from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)
from models import model_builder_video

import utils.misc as misc
import utils.checkpoints as checkpoints
from utils.timer import Timer

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

#------------------------ configures ------------------
length = 8 
stride = 8
width, height = (224, 224)
#------------------------------------------------------


category_table = {
    0: 'celebrate',
    1: 'corner',
    2: 'goal',
    3: 'background',
    4: 'shot',
}

def add_configure():
    cfg.NUM_GPUS = 1
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.TEST.PARAMS_FILE = '../data/checkpoints/run_i3d_nlnet_400k_football/checkpoints/c2_model_iter2500.pkl'
    cfg.VIDEO_DECODER_THREADS = 5
    cfg.TEST.TEST_FULLY_CONV = True

def init_net():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    cfg.TEST.DATA_TYPE = 'test'
    if cfg.TEST.TEST_FULLY_CONV is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 1
    elif cfg.TEST.TEST_FULLY_CONV_FLIP is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 2
    else:
        cfg.TRAIN.CROP_SIZE = 224

    workspace.ResetWorkspace()

    test_model = model_builder_video.ModelBuilder(
        name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
        use_cudnn=True, cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE)
    test_model.build_model()

    workspace.RunNetOnce(test_model.param_init_net)
    net = test_model.net
    checkpoints.load_model_from_params_file_for_test(
            test_model, cfg.TEST.PARAMS_FILE)
    
    # reivse the input blob from `reader_val/reader_test` to new blob that enables frame-sequence input
    clip_blob = core.BlobReference('gpu_0/data')
    net.AddExternalInput(clip_blob)  # insert op into network's head needs to rebuild the network, just add an externalinput blob is enough

    # delete the original video_input_op,  blob('gpu_0/data') is feed by this op before and by hand now
    ops = net.Proto().op
    # assert 'reader' in ops[0].name
    assert ops[0].type == 'CustomizedVideoInput'
    del ops[0]
    workspace.CreateBlob('gpu_0/data')
    
    workspace.CreateNet(net)
    return net

def predict(i3d, clip):
    # clip = np.zeros([1,3,8,224,224]
    clip = (clip - cfg.MODEL.MEAN) / cfg.MODEL.STD
    device_opts = caffe2_pb2.DeviceOption()
    device_opts.device_type = caffe2_pb2.CUDA
    device_opts.cuda_gpu_id = 0
    workspace.FeedBlob('gpu_0/data', clip.astype(np.float32), device_opts)
    
    workspace.RunNet(i3d)
    # data= workspace.FetchBlob('gpu_0/data')
    output_prob = workspace.FetchBlob('gpu_0/softmax')[0]
    pred = output_prob.argmax()
    conf = output_prob[pred]
    return pred, conf 

def collect_clip(cap):
    # stride = cfg['stride']
    # length = cfg['length']
    # width, height = cfg['input_size']

    '''
    clip = np.ndarray(1, [3, length, width, height], dtype=np.float)
    '''
    clip = list()
    frame_index = -1
    while True:
        _, frame = cap.read()
        if not _:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                quit()
            continue
        frame_index += 1
        if not (frame_index+1) % stride == 0:
            continue
        frame = cv2.resize(frame, (width, height))
        '''
        clip[0, 0, (frame_index+1)//stride-1, ...] = frame[..., 0]
        clip[0, 1, (frame_index+1)//stride-1, ...] = frame[..., 1]
        clip[0, 2, (frame_index+1)//stride-1, ...] = frame[..., 2]
        if (frame_index+1)//stride == length:
            return clip
        '''
        clip.append(frame.transpose([2, 0, 1]))
        if len(clip) == length:
            return np.array(clip).reshape([-1, length, 3, width, height]).transpose([0, 2, 1, 3, 4])

def action_recognition(i3d, cap):
    clip = collect_clip(cap)
    pred, conf =  predict(i3d, clip)
    print(pred, conf)


def main():
    config_file = 'deploy_config.yaml'
    cfg_from_file(config_file)
    assert_and_infer_cfg()
    add_configure()

    i3d = init_net()
    cap = cv2.VideoCapture('test.mp4')
    while True:
        action_recognition(i3d, cap)

if __name__ == '__main__':
    main()
