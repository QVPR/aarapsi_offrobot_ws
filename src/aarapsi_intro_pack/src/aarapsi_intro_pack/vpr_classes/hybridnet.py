#!/usr/bin/env python3

import rospkg
import os
os.environ['GLOG_minloglevel'] = '2' # must be done prior to importing caffe to suppress excessive logging.
import caffe
import numpy as np
import cv2
import torch
from tqdm.auto import tqdm

class PlaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, dims=None):
        super().__init__()

        self.from_directory = (isinstance(image_paths, str) or isinstance(image_paths, list))
        self.images         = image_paths
        self.dims           = dims

    def __getitem__(self, index):
        if self.from_directory:
            img = cv2.imread(self.images[index])
        else:
            if self.dims is None: 
                raise Exception('PlaceDataset was initialised with a compressed library without dims')
            img = np.dstack((np.reshape(self.images[index], self.dims),)*3).astype(np.uint8)
        return img, index

    def __len__(self):
        return len(self.images)

class HybridNet_Container:
    def __init__(self, logger=print, cuda=False, 
                 dims=None, target_layer='fc7_new'):
        
        self.cuda           = cuda
        self.logger         = logger
        self.target_layer   = target_layer
        self.dims           = dims
        # keep these features dim fixed as they need to match the network architecture inside "HybridNet"
        self.layerLabs      = ['conv3', 'conv4', 'conv5', 'conv6' ,'pool1', 'pool2', 'fc7_new', 'fc8_new']
        self.layerDims      = [64896, 64896, 43264, 43264, 69984, 43264, 4096, 2543]
        self.layerDict      = dict(zip(self.layerLabs, self.layerDims))

        if self.cuda:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.load()

    def load(self):

        model_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/src/aarapsi_intro_pack/HybridNet/"
        model_file = os.path.join(model_path,'HybridNet.caffemodel')

        self.logger('Loading HybridNet model')

        self.net = caffe.Net(os.path.join(model_path,'deploy.prototxt'), model_file, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(os.path.join(model_path,'amosnet_mean.npy')).mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.net.blobs['data'].reshape(1,3,227,227)

        self.logger('HybridNet loaded')

    def getQryFeat(self, img):
        if isinstance(img, str):
            print(img)
            img = cv2.imread(img)
            if img is None: raise Exception('Image path is invalid')

        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        self.net.forward()
        feat = np.squeeze(self.net.blobs[self.target_layer].data).flatten()
        return feat

    def getRefFeat(self, dataset_input):
        if isinstance(dataset_input, str):
            self.logger("Detected path input: switching mode from compressed libraries to image directory")
            ref_filenames = [os.path.join(dataset_input, filename) 
                             for filename in sorted(os.listdir(dataset_input)) 
                             if os.path.isfile(os.path.join(dataset_input, filename))]
            dataset_clean = PlaceDataset(ref_filenames)
        else:
            dataset_clean = PlaceDataset(dataset_input, dims=self.dims)

        feats = []
        for (input_data, _) in tqdm(dataset_clean):
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', input_data)
            self.net.forward()

            feat = np.squeeze(self.net.blobs[self.target_layer].data).flatten()
            feats.append(feat)
        return np.array(feats)
