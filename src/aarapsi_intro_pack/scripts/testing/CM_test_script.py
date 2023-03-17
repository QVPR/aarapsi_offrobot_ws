#!/usr/bin/env python3

import rospkg
import numpy as np
import os
from aarapsi_intro_pack import aarapsi_NetVLAD_feature_extract
from aarapsi_intro_pack import aarapsi_HybridNet_feature_extract
from PIL import Image
import time
import rospy
import sys
import torch
import torchvision.transforms as transforms
from patchnetvlad.models.models_generic import get_pca_encoding
from aarapsi_intro_pack import Timer

rospy.init_node('test')
#################### Open test image ###################################
#img = Image.open(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/scripts/testing/frame_id_000000.png')
#ref_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/raw_ims/QCR_Lab_3_ref/images/'

img = Image.open(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/scripts/testing/frame_id_000000.png')
ref_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/s1_ccw_o0_e0_a0/forward/'

#################### NetVLAD test example ##############################
# config_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/src/aarapsi_intro_pack/Patch_NetVLAD/patchnetvlad/configs/netvlad_extract.ini'
num_gpus=1
model, config = aarapsi_NetVLAD_feature_extract.load_model(bool(num_gpus), ngpus=num_gpus)

def prep_query_extract(model, cuda, config):
# Somehow, running this much code 'accelerates' the feature_query_extract
# This function when ran first typically takes about 1 second
    device = torch.device("cuda" if cuda else "cpu")
    img = Image.fromarray(np.zeros((1,1,3),dtype=np.uint8))
    the_resize = (int(config['feature_extract']['imageresizeh']), int(config['feature_extract']['imageresizew']))
    mytransform = aarapsi_NetVLAD_feature_extract.input_transform(the_resize)
    input_data = mytransform(img)
    with torch.no_grad():
        input_data = input_data.unsqueeze(dim=0).to(device)
        image_encoding = model.encoder(input_data)
        vlad_global = model.pool(image_encoding)
        vlad_global_pca = get_pca_encoding(model, vlad_global)

t1 = rospy.Time.now().to_sec()
prep_query_extract(model, bool(num_gpus), config)
t2 = rospy.Time.now().to_sec()
rospy.loginfo("Duration: %s" % (str(t2-t1)))
t1 = rospy.Time.now().to_sec()
qry_ftrs = aarapsi_NetVLAD_feature_extract.feature_query_extract(img, model, bool(num_gpus), config)
t2 = rospy.Time.now().to_sec()
rospy.loginfo("Duration: %s" % (str(t2-t1)))
sys.exit()

t1 = rospy.Time.now().to_sec()
ref_ftrs = aarapsi_NetVLAD_feature_extract.feature_ref_extract(ref_path, model, ref_path+'../', bool(num_gpus), config)
t2 = rospy.Time.now().to_sec()
rospy.loginfo("Duration: %s" % (str(t2-t1)))


#################### HybridNet test example ############################
# model, transformer = aarapsi_HybridNet_feature_extract.load_HybridNet()
# t1 = time.time()
# qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model, transformer, np.asarray(img))
# t2 = time.time()
# print(t2-t1)
# print(qry_ftrs.shape)

# t1 = time.time()
# ref_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_ref_extract(model, transformer, ref_path)
# t2 = time.time()
# print(t2-t1)
# print(ref_ftrs.shape)