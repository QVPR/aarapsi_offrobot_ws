#!/usr/bin/env python3

import rospkg
import caffe
import logging
import numpy as np
import scipy.io
import string
import sys
import os
from PIL import Image

def load_HybridNet(mode='cpu'):

    # Disable the Caffe logging output
    logger = logging.getLogger('caffe')
    logger.setLevel(logging.FATAL)

    layerNames = ['conv3', 'conv4', 'conv5', 'conv6' ,'pool1', 'pool2', 'fc7_new', 'fc8_new']
    modelPath = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) +"/src/aarapsi_intro_pack/HybridNet/"
    model_file  = os.path.join(modelPath,'HybridNet.caffemodel')
    layerName = 'fc7_new'

    if mode == 'cpu':
        caffe.set_mode_cpu()
    elif mode == 'gpu':
        caffe.set_mode_gpu()

    pool1_dim = 69984; # 96*27*27
    pool2_dim = 43264;# 256*13*13
    cv3_dim = 64896; # 384*13*13
    cv4_dim = 64896; # 384*13*13
    cv5_dim = 43264; # 256*13*13
    cv6_dim = 43264; # 256*169
    fc7_dim = 4096; # Feature Dimension
    fc8_dim = 2543; # keep these features dim fixed as they need to match the network architecture inside "HybridNet"
    folder = ''
    fileidx = ''
    layerDims = [64896,64896,43264,43264,69984,43264,4096,2543]
    layerDims = dict(zip(layerNames,layerDims))

    print('Loading HybridNet model')

    net = caffe.Net(os.path.join(modelPath,'deploy.prototxt'), model_file, caffe.TEST);

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(os.path.join(modelPath,'amosnet_mean.npy')).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(1,3,227,227)

    print('HybridNet loaded')

    return net, transformer

def HybridNet_query_extract(net, transformer, imData):
    layerName = 'fc7_new'
    net.blobs['data'].data[...] = transformer.preprocess('data', imData)
    out = net.forward()
    fea = np.squeeze(net.blobs[layerName].data)
    return fea.flatten()

def HybridNet_ref_extract(net, transformer, datasetPath):
    layerName = 'fc7_new'

    lst_images = sorted(os.listdir(datasetPath))#[:2000]
    lst_images = [f for f in lst_images if '.png' in f or '.jpg' in f]
    feats = []
    for img in lst_images:
        # imData = caffe.io.load_image(os.path.join(datasetPath,img))#[:800,:,:]
        imData = np.array(Image.open(os.path.join(datasetPath,img)))

        net.blobs['data'].data[...] = transformer.preprocess('data', imData)
        out = net.forward()

        fea = np.squeeze(net.blobs[layerName].data)
        feats.append(fea.flatten())
    return np.array(feats)
