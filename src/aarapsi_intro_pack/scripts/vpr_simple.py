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

# https://answers.ros.org/question/37682/python-deep-copy-of-ros-message/
import copy as copy_module
from copy import deepcopy

class mrc: # main ROS class
    def __init__(self, sub_image_topic, sub_odom_topic):
        self.robot_x            = 0
        self.robot_y            = 0
        self.robot_z            = 0

        rospy.init_node('vpr_simple', anonymous=True)
        rospy.loginfo('Starting vpr_simple node.')

        self.rate_num           = 5 # Hz
        self.rate_obj           = rospy.Rate(self.rate_num)

        self.img_frwd_sub       = rospy.Subscriber(sub_image_topic, CompressedImage, self.img_frwd_callback, queue_size=1) # frwd-view 
        self.odom_sub           = rospy.Subscriber(sub_odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.vpr_feed_pub       = rospy.Publisher("/vpr_simple/image/compressed", CompressedImage, queue_size=1)

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        # flags to denest main loop:
        self.new_img_frwd       = False
        self.new_odom           = False
        self.last_time           = rospy.Time.now()

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
        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame, ftType) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
                feat_list.append(feat)
        else:
            rospy.logerr("Error: No files at reference image path - cannot continue.")
            return [], []
        feat_list = np.array(feat_list)
        return imPath_list, feat_list
    
    def processOdomDataset(self, path):
    # Extract from position .csvs at path, robot x,y,z
    # Return these as nice lists

        x_list, y_list, z_list = [], [], []
        odomPath_list = np.sort(os.listdir(path))
        odomPath_list = [os.path.join(path, f) for f in odomPath_list]
        if len(odomPath_list) > 0:
            for i, odomPath in tqdm(enumerate(odomPath_list)):
                new_odom = np.loadtxt(odomPath, delimiter=',')
                x_list.append(new_odom[0])
                y_list.append(new_odom[1])
                z_list.append(new_odom[2])
        else:
            rospy.logerr("Error: No files at reference odometry path - cannot continue.")
            return [], [], []
        return x_list, y_list, z_list

    def getMatchInds(self, ft_ref, ft_qry, topK=20, metric='euclidean'):
    # top-K matching reference indices for query, shape: [K, N_q]  

        dMat = cdist(ft_ref, ft_qry, metric) # metric: 'euclidean' or 'cosine'
        mInds = np.argsort(dMat, axis=0)[:topK] # shape: K x ft_qry.shape[0]
        return mInds

def vpr_simple():

    #!# Tune Here:
    ftType          = "downsampled_raw" # Feature Type
    K               = 1 # top-K matched reference indices. TODO: currently only works with K=1 (see next TODO in main)
    refImagesPath   = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/rightmerge" # Path where reference images are stored (names are sorted before reading)
    refOdomPath     = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/odo" # Path for where odometry .csv files are stored
    feed_topic      = "/ros_indigosdk_occam/image0/compressed"
    odom_topic      = "/odometry/filtered"

    nmrc = mrc(feed_topic, odom_topic) # make new class instance
    
    # Prepare lists for plotting via plt
    ref_x_data, ref_y_data, _ = nmrc.processOdomDataset(refOdomPath)
    if len(ref_x_data) < 1:
       return # error case; list is empty!

    # Process reference data (only needs to be done once)
    imgList_ref_paths, ft_ref = nmrc.processImageDataset(refImagesPath, ftType)
    if len(imgList_ref_paths) < 1:
       return # error case; list is empty!

    qry_x_data, qry_y_data, _ = [], [], [] # Queries: init as empty

    ## Set up odometry figure:
    # https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
    fig = plt.figure()

    ref_plotted = plt.plot(ref_x_data, ref_y_data, 'b-')[0]
    qry_plotted = plt.plot(qry_x_data, qry_y_data, 'r*')[0]

    plt.title("Odometry Visualised")
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.legend(["Reference", "Query"])

    stretch_x_ref_data = 0.1 * (max(ref_x_data) - min(ref_x_data))
    stretch_y_ref_data = 0.1 * (max(ref_y_data) - min(ref_y_data))
    plt.xlim(min(ref_x_data) - stretch_x_ref_data, max(ref_x_data) + stretch_x_ref_data)
    plt.ylim(min(ref_y_data) - stretch_y_ref_data, max(ref_y_data) + stretch_y_ref_data)
    plt.gca().set_aspect('equal')

    plt.draw()
    #plt.show(block=False)
    #plt.pause(2)

    rospy.loginfo("Reference list processed. Listening for queries...")
    
    # Main loop:
    while not rospy.is_shutdown():

        nmrc.rate_obj.sleep()

        if not (nmrc.new_img_frwd and nmrc.new_odom): # denest
           rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
           continue

        # Grab copy of stored odom + image (protect from overwrite):
        this_odom           = copy_module.deepcopy((nmrc.robot_x, nmrc.robot_y))
        this_image          = copy_module.deepcopy(nmrc.store_img_frwd)

        # Plot new odometry: TODO
        num_queries = len(list(qry_plotted.get_xdata()))
        queries_keep = 10
        if num_queries < queries_keep:
            start_ind = 0
        else:
            start_ind = num_queries - queries_keep + 1
        print((start_ind, num_queries))
        qry_plotted.set_xdata(np.append(qry_plotted.get_xdata()[start_ind:num_queries], this_odom[0]))
        qry_plotted.set_ydata(np.append(qry_plotted.get_ydata()[start_ind:num_queries], this_odom[1]))
        plt.draw()
        plt.pause(0.001)

        # Clear flags:
        nmrc.new_img_frwd   = False
        nmrc.new_odom       = False

        ft_qry = nmrc.getFeat(this_image, ftType).reshape(1,-1) # ensure 2D matrix
        matchInds = nmrc.getMatchInds(ft_ref, ft_qry, K).flatten()[0] # Find match/es

        # TODO: fix multi-match/zero-match problem, K != 1
        try:
            match_img = cv2.imread(imgList_ref_paths[matchInds])
            query_img = cv2.resize(this_image, (match_img.shape[1], match_img.shape[0]), interpolation= cv2.INTER_AREA) # resize to match_img dimensions

            # Black border:
            match_labelled_A = cv2.putText(match_img, "Reference", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(0, 0, 0), thickness=7, lineType=cv2.LINE_AA)
            query_labelled_A = cv2.putText(query_img, "Query", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(0, 0, 0), thickness=7, lineType=cv2.LINE_AA)
            
            # Colour inside:
            match_labelled_B = cv2.putText(match_labelled_A, "Reference", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(100, 255, 100), thickness=2, lineType=cv2.LINE_AA)
            query_labelled_B = cv2.putText(query_labelled_A, "Query", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(100, 255, 100), thickness=2, lineType=cv2.LINE_AA)

            cv2_image_to_pub = np.concatenate((match_labelled_B, query_labelled_B), axis=1)
            
            # Measure timing
            this_time = rospy.Time.now()
            time_diff = this_time - nmrc.last_time
            nmrc.last_time = this_time

            label_string = "Index [%04d] | %2.2f Hz" % (matchInds, 1/time_diff.to_sec())

            # Black border:
            cv2_img_labelled_A = cv2.putText(cv2_image_to_pub, label_string, org=(20, cv2_image_to_pub.shape[0] - 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(0, 0, 0), thickness=7, lineType=cv2.LINE_AA)
            # Colour inside:
            cv2_img_labelled_B = cv2.putText(cv2_img_labelled_A, label_string, org=(20, cv2_image_to_pub.shape[0] - 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                        color=(100, 255, 100), thickness=2, lineType=cv2.LINE_AA)

            ros_image_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(cv2_img_labelled_B, "png")

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