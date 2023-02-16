#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom structures

import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
from enum import Enum
from cv_bridge import CvBridge

import os
from scipy.spatial.distance import cdist
from tqdm import tqdm

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from aarapsi_intro_pack.vpred_tools import *
from aarapsi_intro_pack.vpred_factors import *
from aarapsi_intro_pack import Tolerance_Mode, VPRImageProcessor, FeatureType

class mrc: # main ROS class
    def __init__(self):

        rospy.init_node('vpr_monitor', anonymous=True)
        rospy.loginfo('Starting vpr_monitor node.')

        self.rate_num           = 20.0 # Hz
        self.rate_obj           = rospy.Rate(self.rate_num)

        #!# Tune Here:
        self.PACKAGE_NAME       = 'aarapsi_intro_pack'
        self.FEED_TOPIC         = "/ros_indigosdk_occam/image0/compressed"
        self.ODOM_TOPIC         = "/odometry/filtered"
        self.TOL_MODE           = Tolerance_Mode.METRE_LINE
        self.TOL_THRES          = 5.0
        self.FRAME_ID           = "base_link"
        self.ICON_SIZE          = 50
        self.ICON_DIST          = 20
        self.FEAT_TYPE          = FeatureType.RAW
        self.IMG_DIMS           = (32, 32)
        self.MATCH_METRIC       = 'euclidean'
        self.TIME_HIST_LEN      = 20

        self.REF_DATA_NAME      = "ccw_zeroed_20220208"
        self.CAL_DATA_NAME      = "cw_zeroed_20220208"
        self.DATABASE_PATH      = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/compressed_sets/"

        self.REF_IMG_PATH       = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.REF_DATA_NAME + "/forward_corrected" # Path where real reference images are stored
        self.REF_ODOM_PATH      = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.REF_DATA_NAME + "/odo" # Path for where real reference odometry .csv files are stored

        self.CAL_QRY_PATH       = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.CAL_DATA_NAME + "/forward" # Path where calibration query images are stored
        self.CAL_REF_PATH       = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.CAL_DATA_NAME + "/forward_corrected" # Path where calibration reference images are stored
        self.CAL_ODOM_PATH      = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.CAL_DATA_NAME + "/odo" # Path for where calibration (shared) odometry .csv files are stored

        # Process reference data (only needs to be done once)
        self.ref_ip             = VPRImageProcessor()
        self.ref_info, self.ref_odom = self.ref_ip.npzDatabaseLoadSave(self.DATABASE_PATH, self.REF_DATA_NAME, \
                                                                        self.REF_IMG_PATH, self.REF_ODOM_PATH, \
                                                                        self.FEAT_TYPE, self.IMG_DIMS, do_save=True)

        # Process reference data (only needs to be done once)
        self.cal_ip             = VPRImageProcessor()
        self.cal_info, self.cal_odom = self.cal_ip.npzDatabaseLoadSave(self.DATABASE_PATH, self.CAL_DATA_NAME, \
                                                                        [self.CAL_QRY_PATH, self.CAL_REF_PATH], self.CAL_ODOM_PATH, \
                                                                        self.FEAT_TYPE, self.IMG_DIMS, do_save=True)

def main_loop(nmrc):
    pass

if __name__ == '__main__':
    try:
        nmrc = mrc()

        rospy.loginfo("Reference and calibration lists processed. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass