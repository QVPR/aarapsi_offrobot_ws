#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
import time
from matplotlib import pyplot as plt
from pathlib import Path

import os
from scipy.spatial.distance import cdist
from tqdm import tqdm

class mrc: # main ROS class
    def __init__(self):
        self.robot_x            = 0
        self.robot_y            = 0
        self.robot_z            = 0

        rospy.init_node('vpr_simple', anonymous=True)
        rospy.loginfo('Starting vpr_simple node.')

        self.rate_num           = 5 # Hz
        self.rate_obj           = rospy.Rate(self.rate_num)

        self.img_frwd_sub       = rospy.Subscriber("/ros_indigosdk_occam/image0/compressed", CompressedImage, self.img_frwd_callback, queue_size=1) # frwd-view
        self.odom_sub           = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        self.vpr_feed_pub       = rospy.Publisher("/vpr_simple/image/compressed", CompressedImage, queue_size=1)

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        # flags to denest main loop:
        self.new_img_frwd       = False
        self.new_odom           = False

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.robot_x            = round(msg.pose.pose.position.x, 3)
        self.robot_y            = round(msg.pose.pose.position.y, 3)
        self.robot_z            = round(msg.pose.pose.position.z, 3)
        self.new_odom           = True

    def img_frwd_callback(self, msg):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest forward-facing image received

        self.store_img_frwd     = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_img_frwd       = True

    # TODO: vectorize
    def patchNormaliseImage(self, img, patchLength):
    # take input image, divide into regions, normalise
    # returns: patch normalised image

        img1 = img.astype(float)
        img2 = img1.copy()
        
        if patchLength == 1: # single pixel; already p-n'd
            return img2

        for i in range(img1.shape[0]//patchLength): # floor division -> number of rows
            iStart = i*patchLength
            iEnd = (i+1)*patchLength
            for j in range(img1.shape[1]//patchLength): # floor division -> number of cols
                jStart = j*patchLength
                jEnd = (j+1)*patchLength

                mean1 = np.mean(img1[iStart:iEnd, jStart:jEnd])
                std1 = np.std(img1[iStart:iEnd, jStart:jEnd])

                img2[iStart:iEnd, jStart:jEnd] = img1[iStart:iEnd, jStart:jEnd] - mean1 # offset remove mean
                if std1 == 0:
                    std1 = 0.1
                img2[iStart:iEnd, jStart:jEnd] /= std1 # crush by std

        return img2

    def getFeat(self, im, ftType):
        im = cv2.resize(im, (64, 64))
        ft = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if ftType == "downsampled_raw":
            pass # already done
        elif ftType == "downsampled_patchNorm":
            ft = self.patchNormaliseImage(ft, 8)
        return ft.flatten()

    def processImageDataset(self, path, ftType): 
    # Extract images and their features from path
    # Store in arrays and return them.

        imPath_list = np.sort(os.listdir(path))
        imPath_list = [os.path.join(path, f) for f in imPath_list]
        feat_list = []
        for i, imPath in tqdm(enumerate(imPath_list)):
            frame = cv2.imread(imPath)[:, :, ::-1]
            feat = self.getFeat(frame, ftType) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
            feat_list.append(feat)
        feat_list = np.array(feat_list)
        return imPath_list, feat_list

    def getMatchInds(self, ft_ref, ft_qry, topK=20, metric='euclidean'):
    # top-K matching reference indices for query, shape: [K, N_q]  

        dMat = cdist(ft_ref, ft_qry, metric) # metric: 'euclidean' or 'cosine'
        mInds = np.argsort(dMat, axis=0)[:topK] # shape: K x ft_qry.shape[0]
        return mInds

def vpr_simple():

    nmrc = mrc() # make new class instance

    #!# Tune Here:
    ftType  = "downsampled_raw" # Feature Type
    K       = 1 # top-K matched reference indices. TODO: currently only works with K=1 (see next TODO in main)

    # Path where reference images are stored (names are sorted before reading)
    refImagesPath = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/vpr_primer"

    # Process reference data (only needs to be done once)
    imgList_ref_paths, ft_ref = nmrc.processImageDataset(refImagesPath, ftType)
    
    # Main loop:
    while not rospy.is_shutdown():

        nmrc.rate_obj.sleep()

        if not (nmrc.new_img_frwd and nmrc.new_odom): # denest
           continue

        ft_qry = nmrc.getFeat(nmrc.store_img_frwd, ftType)
        matchInds = nmrc.getMatchInds(ft_ref, ft_qry, K) # Find match/es

        # TODO: fix multi-match/zero-match problem, K != 1
        try:
            match_img = cv2.imread(imgList_ref_paths[matchInds])[:, :, ::-1]
            query_img = nmrc.new_img_frwd

            cv2_image_to_pub = np.concatenate((match_img, query_img), axis=1)
            ros_image_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(cv2_image_to_pub, "png")

            nmrc.vpr_feed_pub.publish(ros_image_to_pub)
        except Exception as e:
            rospy.logerr("Error caught in ROS msg processing. K condition violation likely. Caught error:")
            rospy.logwarn(e)


if __name__ == '__main__':
    try:
        vpr_simple()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass