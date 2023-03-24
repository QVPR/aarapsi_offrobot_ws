#!/usr/bin/env python3

import rospkg
import numpy as np
import os
import time
import rospy
import torch
from aarapsi_intro_pack import VPRImageProcessor

gpu_exists=torch.cuda.is_available()
rospy.init_node('test')
#################### Open test data ###################################

fn = 's1_ccw_o0_e0_a0'
dims = (64, 64)
cam = 'forward'

dbp = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/compressed_sets'
ref_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/' + fn + '/' + cam

ip = VPRImageProcessor()
ip.npzLoader(dbp, fn, dims)
ind = int(np.random.rand() * (len(ip.SET_DICT['times']) - 1))
ft_raw = ip.SET_DICT['img_feats']['RAW'][cam]
img = np.dstack((np.reshape(ft_raw[ind], dims),)*3).astype(np.uint8)

#################### NetVLAD test example ##############################

from aarapsi_intro_pack.aarapsi_NetVLAD_feature_extract import NetVLAD_Container

netvlad_obj = NetVLAD_Container(cuda=gpu_exists, ngpus=int(gpu_exists), logger=lambda x: (x), dims=dims)

t1 = rospy.Time.now().to_sec()
qry_ftrs = netvlad_obj.feature_query_extract(img)
t2 = rospy.Time.now().to_sec()
rospy.loginfo("Duration: %s" % (str(t2-t1)))

t1 = rospy.Time.now().to_sec()
#ref_ftrs = netvlad_obj.feature_ref_extract(ref_path) # via path 
ref_ftrs = netvlad_obj.feature_ref_extract(ft_raw) # via compressed library
print((ref_ftrs.shape, type(ref_ftrs.flatten()[0])))
t2 = rospy.Time.now().to_sec()
rospy.loginfo("Duration: %s" % (str(t2-t1)))
    

#################### HybridNet test example ############################
os.environ['GLOG_minloglevel'] = '2' # must be done prior to importing caffe to suppress excessive logging.
from aarapsi_intro_pack import aarapsi_HybridNet_feature_extract

model, transformer = aarapsi_HybridNet_feature_extract.load_HybridNet(mode=('gpu' if gpu_exists else 'cpu'))
t1 = time.time()
qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model, transformer, np.asarray(img))
t2 = time.time()
print(t2-t1)
print(qry_ftrs.shape)
t1 = time.time()
qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model, transformer, np.asarray(img))
t2 = time.time()
print(t2-t1)
print(qry_ftrs.shape)

# t1 = time.time()
# qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model_gpu, transformer_gpu, np.asarray(img))
# t2 = time.time()
# print(t2-t1)
# print(qry_ftrs.shape)
# t1 = time.time()
# qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model_gpu, transformer_gpu, np.asarray(img))
# t2 = time.time()
# print(t2-t1)
# print(qry_ftrs.shape)

# t1 = time.time()
# ref_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_ref_extract(model, transformer, ref_path)
# t2 = time.time()
# print(t2-t1)
# print(ref_ftrs.shape)