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

    def getMatchInds(self, ft_ref, ft_qry, metric='euclidean'):
    # top matching reference index for query

        dMat = cdist(ft_ref, ft_qry, metric) # metric: 'euclidean' or 'cosine'
        mInds = np.argsort(dMat, axis=0)[:1] # shape: K x ft_qry.shape[0]
        return mInds.flatten()[0], dMat
    
    def getTrueInds(self, real_odom, x_data, y_data):
    # Compare measured odometry to reference odometry and find best match

        squares = np.square(np.array(x_data) - real_odom[0]) + np.square(np.array(y_data) - real_odom[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

def labelImage(img_in, textstring, org_in, colour):
    # Black border:
    img_A = cv2.putText(img_in, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=(0,0,0), thickness=7, lineType=cv2.LINE_AA)
    # Colour inside:
    img_B = cv2.putText(img_A, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=colour, thickness=2, lineType=cv2.LINE_AA)
    return img_B

def vpr_simple():

    #!# Tune Here:
    ftType          = "downsampled_raw" # Feature Type
    refImagesPath   = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/rightmerge" # Path where reference images are stored (names are sorted before reading)
    refOdomPath     = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/odo" # Path for where odometry .csv files are stored
    feed_topic      = "/ros_indigosdk_occam/image0/compressed"
    odom_topic      = "/odometry/filtered"

    nmrc = mrc(feed_topic, odom_topic) # make new class instance
    
    ## Prepare lists for plotting via plt
    # Reference: Full list for use in query and true
    # Match: the odom for the matched image
    # True: for n'th index, the correct odom from ROS topic
    ref_x_data, ref_y_data, _ = nmrc.processOdomDataset(refOdomPath)
    if len(ref_x_data) < 1:
       return # error case; list is empty!

    # Process reference data (only needs to be done once)
    imgList_ref_paths, ft_ref = nmrc.processImageDataset(refImagesPath, ftType)
    if len(imgList_ref_paths) < 1:
       return # error case; list is empty!

    ## Set up odometry figure:
    # https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig, axes = plt.subplots(1, 2)

    plt.sca(axes[0]) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=5)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'go', markersize=5)[0] # true image (correct match)
    axes[0].set(xlabel='Index', ylabel='Distance')
    axes[0].legend(["Image Distances", "Selected", "True"])
    axes[0].set_xlim(0, len(ref_x_data))
    axes[0].set_ylim(0, 10000)

    plt.sca(axes[1])
    ref_plotted = plt.plot(ref_x_data, ref_y_data, 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    fig.suptitle("Odometry Visualised")
    axes[1].set(xlabel='X-Axis', ylabel='Y-Axis')
    axes[1].legend(["Reference", "Match", "True"])

    stretch_x_ref_data = 0.1 * (max(ref_x_data) - min(ref_x_data))
    stretch_y_ref_data = 0.1 * (max(ref_y_data) - min(ref_y_data))

    axes[1].set_xlim(min(ref_x_data) - stretch_x_ref_data, max(ref_x_data) + stretch_x_ref_data)
    axes[1].set_ylim(min(ref_y_data) - stretch_y_ref_data, max(ref_y_data) + stretch_y_ref_data)
    axes[1].set_aspect('equal')

    fig.show()

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

        # Clear flags:
        nmrc.new_img_frwd   = False
        nmrc.new_odom       = False

        ft_qry = nmrc.getFeat(this_image, ftType).reshape(1,-1) # ensure 2D matrix
        matchInd, dist_vector_calc = nmrc.getMatchInds(ft_ref, ft_qry) # Find match/es
        trueInd = nmrc.getTrueInds(this_odom, ref_x_data, ref_y_data) # find correct match based on shortest difference to measured odometry

        # Plot new odometry:
        num_queries = len(list(tru_plotted.get_xdata()))
        queries_keep = 10
        if num_queries < queries_keep:
            start_ind = 0
        else:
            start_ind = num_queries - queries_keep + 1

        ## x,y plot:
        # Append new value for "match" (what it matched the image to)
        mat_plotted.set_xdata(np.append(mat_plotted.get_xdata()[start_ind:num_queries], ref_x_data[matchInd]))
        mat_plotted.set_ydata(np.append(mat_plotted.get_ydata()[start_ind:num_queries], ref_y_data[matchInd]))
        # Append new value for "true" (what it should be from the robot odom)
        tru_plotted.set_xdata(np.append(tru_plotted.get_xdata()[start_ind:num_queries], ref_x_data[trueInd]))
        tru_plotted.set_ydata(np.append(tru_plotted.get_ydata()[start_ind:num_queries], ref_y_data[trueInd]))

        ## distance vector plot:
        # overwrite with new distance vector / image distance:
        dist_vector.set_xdata(range(len(dist_vector_calc-1)))
        dist_vector.set_ydata(dist_vector_calc)
        # overwrite with new lowest match:
        lowest_dist.set_xdata(matchInd)
        lowest_dist.set_ydata(dist_vector_calc[matchInd])
        # overwrite with new truth value:
        actual_dist.set_xdata(trueInd)
        actual_dist.set_ydata(dist_vector_calc[trueInd])
        
        fig.show()
        plt.pause(0.001)

        try:
            match_img = cv2.imread(imgList_ref_paths[matchInd])
            query_img = cv2.resize(this_image, (match_img.shape[1], match_img.shape[0]), interpolation= cv2.INTER_AREA) # resize to match_img dimensions
            
            match_img_lab = labelImage(match_img, "Reference", (20,40), (100,255,100))
            query_img_lab = labelImage(query_img, "Query", (20,40), (100,255,100))

            cv2_image_to_pub = np.concatenate((match_img_lab, query_img_lab), axis=1)
            
            # Measure timing
            this_time = rospy.Time.now()
            time_diff = this_time - nmrc.last_time
            nmrc.last_time = this_time

            label_string = "Index [%04d] @ %2.2f Hz" % (matchInd, 1/time_diff.to_sec())
            cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

            ros_image_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(cv2_img_lab, "png")
            nmrc.vpr_feed_pub.publish(ros_image_to_pub)

        except Exception as e:
           rospy.logerr("Error caught in ROS msg processing. K condition violation likely (K = 0?). Caught error:")
           rospy.logwarn(e)


if __name__ == '__main__':
    try:
        vpr_simple()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass