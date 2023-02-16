#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
from enum import Enum

import os
from scipy.spatial.distance import cdist
from tqdm import tqdm
from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom structures

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from aarapsi_intro_pack.vpred_tools import *
from aarapsi_intro_pack.vpred_factors import *
from aarapsi_intro_pack import Tolerance_Mode

class mrc: # main ROS class
    def __init__(self):

        rospy.init_node('vpr_monitor', anonymous=True)
        rospy.loginfo('Starting vpr_monitor node.')

        self.rate_num        = 20.0 # Hz
        self.rate_obj        = rospy.Rate(self.rate_num)

        #!# Tune Here:
        self.PACKAGE_NAME    = 'aarapsi_intro_pack'
        self.FEED_TOPIC      = "/ros_indigosdk_occam/image0/compressed"
        self.ODOM_TOPIC      = "/odometry/filtered"
        self.TOL_MODE        = Tolerance_Mode.METRE_LINE
        self.TOL_THRES       = 5.0
        self.FRAME_ID        = "base_link"
        self.ICON_SIZE       = 50
        self.ICON_DIST       = 20
        self.IMG_DIMS        = (32, 32)
        self.MATCH_METRIC    = 'euclidean'
        self.TIME_HIST_LEN   = 20

        self.REF_IMG_PATH    = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/ccw_loop/forward" # Path where reference images are stored
        self.REF_ODOM_PATH   = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/ccw_loop/odo" # Path for where reference odometry .csv files are stored

        self.CAL_IMG_PATH    = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/cw_loop/forward" # Path where calibration images are stored
        self.CAL_ODOM_PATH   = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/cw_loop/odo" # Path for where calibration odometry .csv files are stored

def main_loop(nmrc):
    pass

if __name__ == '__main__':
    try:
        nmrc = mrc()

        rospy.loginfo("Reference list processed. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass