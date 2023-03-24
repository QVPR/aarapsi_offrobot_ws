#!/usr/bin/env python3

'''
Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Extracts Patch-NetVLAD local and NetVLAD global features from a given directory of images.

Configuration settings are stored in configs folder, with compute heavy performance or light-weight alternatives
available.

Features are saved into a nominated output directory, with one file per image per patch size.

Code is dynamic and can be configured with essentially *any* number of patch sizes, by editing the config files.
'''

import configparser
import os
from os.path import join, exists, isfile
from os import makedirs

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
from download_models import download_all_models

class PlaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform, dims=None):
        super().__init__()

        self.from_directory = (isinstance(image_paths, str) or isinstance(image_paths, list))
        self.images         = image_paths
        self.transform      = transform
        self.dims           = dims

    def __getitem__(self, index):
        if self.from_directory:
            img     = Image.open(self.images[index])
        else:
            if self.dims is None: 
                raise Exception('PlaceDataset was initialised with a compressed library without dims')
            np_img  = np.dstack((np.reshape(self.images[index], self.dims),)*3).astype(np.uint8)
            img     = Image.fromarray(np_img)
        return self.transform(img), index

    def __len__(self):
        return len(self.images)

class NetVLAD_Container:
    def __init__(self, logger=print, cuda=False, ngpus=0, 
                 dims=None, imw=640, imh=480, 
                 batchsize=5, cachebatchsize=5, num_pcs=4096, 
                 threads=0, resumepath='./pretrained_models/mapillary_WPCA'):
        
        self.cuda           = cuda
        self.ngpus          = ngpus
        self.logger         = logger
        self.dims           = dims
        self.imw            = imw
        self.imh            = imh
        self.batchsize      = batchsize
        self.cachebatchsize = cachebatchsize
        self.num_pcs        = num_pcs
        self.threads        = threads
        self.resumepath     = resumepath
        self.transform      = self.input_transform()

        self.load()
        self.prep()


    def load(self):
        self.config = configparser.ConfigParser()
        self.config['feature_extract'] = {'batchsize': self.batchsize, 'cachebatchsize': self.cachebatchsize, 
                                    'imageresizew': self.imw, 'imageresizeh': self.imh}
        self.config['global_params'] = {'pooling': 'netvlad', 'resumepath': self.resumepath, 
                                'threads': self.threads, 'num_pcs': self.num_pcs, 'ngpu': self.ngpus}

        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found")
        self.device = torch.device("cuda" if self.cuda else "cpu")

        encoder_dim, encoder = get_backend()

        # must resume to do extraction
        if int(self.config['global_params']['num_pcs']) > 0:
            resume_ckpt = self.config['global_params']['resumePath'] + self.config['global_params']['num_pcs'] + '.pth.tar'
        else:
            resume_ckpt = self.config['global_params']['resumePath'] + '.pth.tar'

        # backup: try whether resume_ckpt is relative to PATCHNETVLAD_ROOT_DIR
        if not isfile(resume_ckpt):
            resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
            if not isfile(resume_ckpt):
                download_all_models(ask_for_permission=True)

        if isfile(resume_ckpt):
            self.logger("=> Trying to load checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            if bool(self.num_pcs):
                assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(self.config['global_params']['num_pcs'])
            self.config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            self.model = get_model(encoder, encoder_dim, self.config['global_params'], append_pca_layer=bool(self.num_pcs))
            self.model.load_state_dict(checkpoint['state_dict'])
            
            if int(self.config['global_params']['ngpu']) > 1 and torch.cuda.device_count() > 1:
                self.model.encoder = torch.nn.DataParallel(self.model.encoder)
                self.model.pool = torch.nn.DataParallel(self.model.pool)
        
            self.model = self.model.to(self.device)
            self.logger("=> Successfully loaded checkpoint '{}'".format(resume_ckpt, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))
        self.model.eval()

    def prep(self):
    # Somehow, running this much code 'accelerates' the feature_query_extract
    # This function when ran first typically takes about 1 second
        input_data = self.transform(Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)))
        with torch.no_grad():
            input_data = input_data.unsqueeze(dim=0).to(self.device)
            image_encoding = self.model.encoder(input_data)
            vlad_global = self.model.pool(image_encoding)
            get_pca_encoding(self.model, vlad_global)

    def input_transform(self):
        return transforms.Compose([
            transforms.Resize((self.imh, self.imw)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def getQryFeat(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        input_data = self.transform(img)

        with torch.no_grad():
            input_data = input_data.unsqueeze(dim=0).to(self.device)
            image_encoding = self.model.encoder(input_data)
            vlad_global = self.model.pool(image_encoding)
            vlad_global_pca = get_pca_encoding(self.model, vlad_global)

        return np.squeeze(vlad_global_pca.detach().cpu().numpy())

    def getRefFeat(self, dataset_input, save_dir=None):
        if isinstance(dataset_input, str):
            self.logger("Detected path input: switching mode from compressed libraries to image directory")
            ref_filenames = [os.path.join(dataset_input, filename) 
                             for filename in sorted(os.listdir(dataset_input)) 
                             if os.path.isfile(os.path.join(dataset_input, filename))]
            dataset_clean = PlaceDataset(ref_filenames, self.transform)
        else:
            dataset_clean = PlaceDataset(dataset_input, self.transform, dims=self.dims)

        dataLoader  = DataLoader(dataset     = dataset_clean, 
                                 num_workers = int(self.config['global_params']['threads']),
                                 batch_size  = int(self.config['feature_extract']['cacheBatchSize']),
                                 shuffle     = False, 
                                 pin_memory  = self.cuda)

        self.model.eval()
        with torch.no_grad():
            self.logger('====> Extracting Features')
            db_feat = np.empty((len(dataset_clean), int(self.config['global_params']['num_pcs'])), dtype=np.float32)

            for (input_data, indices) in tqdm(dataLoader): # manage batches and threads
                indices_np              = indices.detach().numpy()
                input_data              = input_data.to(self.device)
                image_encoding          = self.model.encoder(input_data)
                vlad_global             = self.model.pool(image_encoding)
                vlad_global_pca         = get_pca_encoding(self.model, vlad_global)
                db_feat[indices_np, :]  = vlad_global_pca.detach().cpu().numpy()

        if not (save_dir is None):
            if not exists(save_dir):
                makedirs(save_dir)
            output_global_features_filename = join(save_dir, 'NetVLAD_feats.npy')
            np.save(output_global_features_filename, db_feat)

        return db_feat