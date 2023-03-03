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


import argparse
import configparser
import os
from os.path import join, exists, isfile
from os import makedirs

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from patchnetvlad.tools.datasets import PlaceDataset
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def feature_query_extract(img, model, cuda, config):
    device = torch.device("cuda" if cuda else "cpu")

    the_resize = (int(config['feature_extract']['imageresizeh']), int(config['feature_extract']['imageresizew']))
    mytransform = input_transform(the_resize)
    input_data = mytransform(img)

    with torch.no_grad():
        input_data = input_data.unsqueeze(dim=0).to(device)
        image_encoding = model.encoder(input_data)
        vlad_global = model.pool(image_encoding)
        vlad_global_pca = get_pca_encoding(model, vlad_global)

    return np.squeeze(vlad_global_pca.detach().cpu().numpy())

def feature_ref_extract(dataset_file_path, dataset_root_dir, model, device, output_features_dir, cuda, config):
    if not os.path.isfile(dataset_file_path):
        dataset_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', dataset_file_path)

    dataset = PlaceDataset(None, dataset_file_path, dataset_root_dir, None, config['feature_extract'])

    if not exists(output_features_dir):
        makedirs(output_features_dir)

    output_local_features_prefix = join(output_features_dir, 'patchfeats')
    output_global_features_filename = join(output_features_dir, 'globalfeats.npy')

    pool_size = int(config['global_params']['num_pcs'])

    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['feature_extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=(cuda))

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        db_feat = np.empty((len(eval_set), pool_size), dtype=np.float32)

        for iteration, (input_data, indices) in \
                enumerate(tqdm(test_data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            image_encoding = model.encoder(input_data)
            vlad_global = model.pool(image_encoding)
            vlad_global_pca = get_pca_encoding(model, vlad_global)
            db_feat[indices_np, :] = vlad_global_pca.detach().cpu().numpy()

    np.save(output_global_features_filename, db_feat)

    return db_feat


def load_model(configfile, cuda):
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    if config['global_params']['num_pcs'] != '0':
        resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'
    else:
        resume_ckpt = config['global_params']['resumePath'] + '.pth.tar'

    # backup: try whether resume_ckpt is relative to PATCHNETVLAD_ROOT_DIR
    if not isfile(resume_ckpt):
        resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
        if not isfile(resume_ckpt):
            from download_models import download_all_models
            download_all_models(ask_for_permission=True)

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        if config['global_params']['num_pcs'] != '0':
            assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        if config['global_params']['num_pcs'] != '0':
            use_pca = True
        else:
            use_pca = False
        model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=use_pca)
        model.load_state_dict(checkpoint['state_dict'])
        
        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            model.pool = nn.DataParallel(model.pool)
       
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))
    model.eval()

    return model, config