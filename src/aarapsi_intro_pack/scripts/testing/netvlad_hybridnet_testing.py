#!/usr/bin/env python3

import rospkg
import numpy as np
import os
import time
import rospy
import torch
from aarapsi_intro_pack import VPRImageProcessor
from aarapsi_intro_pack.vpr_classes import NetVLAD_Container, HybridNet_Container

gpu_exists=torch.cuda.is_available()
rospy.init_node('test', log_level=rospy.DEBUG)
#################### Open test data ###################################

fn = 's1_ccw_o0_e0_a0'
dims = (64, 64)
cam = 'forward'
do_netvlad = True
do_hybridnet = True

dbp = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/compressed_sets'
ref_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/' + fn + '/' + cam
img_path = ref_path + '/frame_id_000000.png'

ip = VPRImageProcessor()
ip.npzLoader(dbp, fn, dims)
ind = int(np.random.rand() * (len(ip.SET_DICT['times']) - 1))
ft_raw = ip.SET_DICT['img_feats']['RAW'][cam]
img = np.dstack((np.reshape(ft_raw[ind], dims),)*3).astype(np.uint8)

#################### NetVLAD test example ##############################
if do_netvlad:
    netvlad_obj = NetVLAD_Container(cuda=gpu_exists, ngpus=int(gpu_exists), logger=lambda x: (x), dims=dims)

    t1 = rospy.Time.now().to_sec()
    qry_ftrs = netvlad_obj.getQryFeat(img_path)
    t2 = rospy.Time.now().to_sec()
    qry_ftrs = netvlad_obj.getQryFeat(img)
    t3 = rospy.Time.now().to_sec()
    rospy.logdebug("Durations: %s, %s" % (str(t2-t1), str(t3-t2)))

    t1 = rospy.Time.now().to_sec()
    ref_ftrs = netvlad_obj.getRefFeat(ref_path) # via path 
    t2 = rospy.Time.now().to_sec()
    ref_ftrs = netvlad_obj.getRefFeat(ft_raw) # via compressed library
    t3 = rospy.Time.now().to_sec()
    rospy.logdebug("Durations: %s, %s" % (str(t2-t1), str(t3-t2)))
    
#################### HybridNet test example ############################
if do_hybridnet:
    hybridnet_obj = HybridNet_Container(cuda=gpu_exists, logger=lambda x: (x), dims=dims)

    t1 = rospy.Time.now().to_sec()
    qry_ftrs = hybridnet_obj.getQryFeat(img_path)
    t2 = rospy.Time.now().to_sec()
    qry_ftrs = hybridnet_obj.getQryFeat(img)
    t3 = rospy.Time.now().to_sec()
    rospy.logdebug("Durations: %s, %s" % (str(t2-t1), str(t3-t2)))

    t1 = rospy.Time.now().to_sec()
    ref_ftrs = hybridnet_obj.getRefFeat(ref_path) # via path 
    t2 = rospy.Time.now().to_sec()
    ref_ftrs = hybridnet_obj.getRefFeat(ft_raw) # via compressed library
    t3 = rospy.Time.now().to_sec()
    rospy.logdebug("Durations: %s, %s" % (str(t2-t1), str(t3-t2)))