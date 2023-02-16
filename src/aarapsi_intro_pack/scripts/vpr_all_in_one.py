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

from scipy.spatial.distance import cdist
from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom structures
from aarapsi_intro_pack import VPRImageProcessor, Tolerance_Mode, labelImage, makeImage, \
                                doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig

class mrc: # main ROS class
    def __init__(self):

        rospy.init_node('vpr_all_in_one', anonymous=True)
        rospy.loginfo('Starting vpr_all_in_one node.')

        self.rate_num        = 20.0 # Hz, maximum of between 21-26 Hz (varies) with no plotting/image/ground truth/compression.
        self.rate_obj        = rospy.Rate(self.rate_num)

        #!# Tune Here:
        self.FEAT_TYPE       = "downsampled_raw" # Feature Type
        self.IMG_DIMS        = (32, 32)
        self.PACKAGE_NAME    = 'aarapsi_intro_pack'
        self.REF_DATA_NAME   = "ccw_loop"
        self.DATABASE_PATH   = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/compressed_sets/"
        # Path where reference images are stored (names are sorted before reading):
        self.REF_IMG_PATH    = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.REF_DATA_NAME + "/forward"
        # Path for where odometry .csv files are stored: 
        self.REF_ODOM_PATH   = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/" + self.REF_DATA_NAME + "/odo" 
        self.FEED_TOPIC      = "/ros_indigosdk_occam/image0/compressed"
        self.ODOM_TOPIC      = "/odometry/filtered"
        self.TOL_MODE        = Tolerance_Mode.METRE_LINE
        self.TOL_THRES       = 5.0
        self.FRAME_ID        = "base_link"
        self.ICON_SIZE       = 50
        self.ICON_DIST       = 20
        self.MATCH_METRIC    = 'euclidean'
        self.TIME_HIST_LEN   = 20

        #!# Enable/Disable Features (Label topic will always be generated):
        self.DO_COMPRESS     = False
        self.DO_PLOTTING     = False
        self.MAKE_IMAGE      = False
        self.GROUND_TRUTH    = False

        self.ego                = [0.0, 0.0, 0.0] # robot position

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        self.img_frwd_sub       = rospy.Subscriber(self.FEED_TOPIC, CompressedImage, self.img_frwd_callback, queue_size=1) # frwd-view 
        self.odom_sub           = rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)

        self.img_tpc_mode       = ""
        self.image_type         = Image
        self.label_type         = ImageLabelStamped

        if self.DO_COMPRESS:
            self.img_tpc_mode   = "/compressed"
            self.image_type     = CompressedImage
            self.label_type     = CompressedImageLabelStamped

        if self.MAKE_IMAGE:
            self.vpr_feed_pub   = rospy.Publisher("/vpr_simple/image" + self.img_tpc_mode, self.image_type, queue_size=1)
        self.vpr_label_pub      = rospy.Publisher("/vpr_simple/image/label" + self.img_tpc_mode, self.label_type, queue_size=1)

        if self.DO_PLOTTING:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15,4))
            self.timer_plot     = rospy.Timer(rospy.Duration(0.1), self.timer_plot_callback) # 10 Hz

        # flags to denest main loop:
        self.new_query          = False
        self.new_odom           = False
        self.do_show            = False
        self.main_ready         = False

        self.last_time          = rospy.Time.now()
        self.time_history       = []

        # Process reference data (only needs to be done once)
        self.image_processor    = VPRImageProcessor()
        self.ref_info, self.ref_odom = self.image_processor.npzDatabaseLoadSave(self.DATABASE_PATH, self.REF_DATA_NAME, \
                                                                                self.REF_IMG_PATH, self.REF_ODOM_PATH, \
                                                                                self.FEAT_TYPE, self.IMG_DIMS, do_save=True)

        if self.DO_PLOTTING:
            # Prepare figures:
            self.fig.suptitle("Odometry Visualised")
            self.fig_mtrx_handles = doMtrxFig(self.axes[0], self.ref_odom) # Make simularity matrix figure
            self.fig_dvec_handles = doDVecFig(self.axes[1], self.ref_odom) # Make distance vector figure
            self.fig_odom_handles = doOdomFig(self.axes[2], self.ref_odom) # Make odometry figure
            self.fig.show()

        self.ICON_DICT = {'size': self.ICON_SIZE, 'dist': self.ICON_DIST, 'icon': [], 'good': [], 'poor': []}
        # Load icons:
        if self.MAKE_IMAGE:
            good_icon = cv2.imread(rospkg.RosPack().get_path(self.PACKAGE_NAME) + '/media/' + 'tick.png', cv2.IMREAD_UNCHANGED)
            poor_icon = cv2.imread(rospkg.RosPack().get_path(self.PACKAGE_NAME) + '/media/' + 'cross.png', cv2.IMREAD_UNCHANGED)
            self.ICON_DICT['good'] = cv2.resize(good_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)
            self.ICON_DICT['poor'] = cv2.resize(poor_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)

        self.main_timer = rospy.Timer(rospy.Duration(1/self.rate_num), self.main_cb)

    def main_cb(self, event):
        self.main_ready = True

    def timer_plot_callback(self, event):
    # Toggle flag so that visualisation is performed at a lower rate than main loop processing (10Hz instead of >>10Hz)

        self.do_show            = True

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.ego                = [round(msg.pose.pose.position.x, 3), round(msg.pose.pose.position.y, 3), round(msg.pose.pose.position.z, 3)]
        self.new_odom           = True

    def img_frwd_callback(self, msg):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest forward-facing image received

        self.store_query        = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_query          = True
        
    def getMatchInd(self, ft_qry, metric='euclidean'):
    # top matching reference index for query

        dMat = cdist(self.ref_info['fts'], ft_qry, metric) # metric: 'euclidean' or 'cosine'
        mInd = np.argmin(dMat[:])
        return mInd, dMat
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match

        squares = np.square(np.array(self.ref_odom['x']) - self.ego[0]) + \
                            np.square(np.array(self.ref_odom['y']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, cv2_img, fid, tInd, mInd, dvc, img_path):
        if self.DO_COMPRESS:
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpg") # jpg (png slower)
            struct_to_pub = CompressedImageLabelStamped()
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "passthrough")
            struct_to_pub = ImageLabelStamped()
            
        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = fid

        struct_to_pub.data.queryImage = ros_image_to_pub
        struct_to_pub.data.trueId = tInd
        struct_to_pub.data.matchId = mInd
        struct_to_pub.data.matchPath = img_path
        struct_to_pub.data.odom.x = self.ego[0]
        struct_to_pub.data.odom.y = self.ego[1]
        struct_to_pub.data.odom.z = self.ego[2]
        struct_to_pub.data.dvc = dvc
        struct_to_pub.header.frame_id = fid
        struct_to_pub.header.stamp = rospy.Time.now()

        if self.MAKE_IMAGE:
            self.vpr_feed_pub.publish(ros_image_to_pub)
        self.vpr_label_pub.publish(struct_to_pub)

def main_loop(nmrc):
# Main loop process

    if nmrc.do_show and nmrc.DO_PLOTTING:
        nmrc.fig.canvas.draw()
        nmrc.do_show = False

    if not (nmrc.new_query and nmrc.new_odom and nmrc.main_ready): # denest
        #rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    # Clear flags:
    nmrc.new_query      = False
    nmrc.new_odom       = False
    nmrc.main_ready     = False

    ft_qry = nmrc.image_processor.getFeat(nmrc.store_query, size=2)
    matchInd, dvc = nmrc.getMatchInd(ft_qry, nmrc.MATCH_METRIC) # Find match
    trueInd = -1 #default; can't be negative.
    if nmrc.GROUND_TRUTH:
        trueInd = nmrc.getTrueInd() # find correct match based on shortest difference to measured odometry
    else:
        nmrc.ICON_DICT['size'] = -1

    ground_truth_string = ""
    if nmrc.GROUND_TRUTH and nmrc.MAKE_IMAGE:
        # Determine if we are within tolerance:
        nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['poor']
        if nmrc.TOL_MODE == Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(nmrc.ref_odom['x'][trueInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_odom['y'][trueInd] - nmrc.ego[1])) 
            tolString = "MCT"
        elif nmrc.TOL_MODE == Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(nmrc.ref_odom['x'][matchInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_odom['y'][matchInd] - nmrc.ego[1])) 
            tolString = "MCM"
        elif nmrc.TOL_MODE == Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(nmrc.ref_odom['x'][trueInd] - nmrc.ref_odom['x'][matchInd]) + \
                    np.square(nmrc.ref_odom['y'][trueInd] - nmrc.ref_odom['y'][matchInd])) 
            tolString = "ML"
        elif nmrc.TOL_MODE == Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
            tolString = "F"

        if tolError < nmrc.TOL_THRES:
            nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['good']

        ground_truth_string = ", Error: %2.2f%s" % (tolError, tolString)

    if nmrc.DO_PLOTTING:
        # Update odometry visualisation:
        updateMtrxFig(matchInd, trueInd, dvc, nmrc.ref_odom, nmrc.fig_mtrx_handles)
        updateDVecFig(matchInd, trueInd, dvc, nmrc.ref_odom, nmrc.fig_dvec_handles)
        updateOdomFig(matchInd, trueInd, dvc, nmrc.ref_odom, nmrc.fig_odom_handles)

    if nmrc.MAKE_IMAGE:
        cv2_image_to_pub = makeImage(nmrc.store_query, nmrc.ref_info['paths'][matchInd], \
                                        nmrc.ICON_DICT['icon'], icon_size=nmrc.ICON_DICT['size'], icon_dist=nmrc.ICON_DICT['dist'])
        # Measure timing
        this_time = rospy.Time.now()
        time_diff = this_time - nmrc.last_time
        nmrc.last_time = this_time
        nmrc.time_history.append(time_diff.to_sec())
        num_time = len(nmrc.time_history)
        if num_time > nmrc.TIME_HIST_LEN:
            nmrc.time_history.pop(0)
        time_average = sum(nmrc.time_history) / num_time

        label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + ground_truth_string
        cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

        img_to_pub = cv2_img_lab
    
    else:
        img_to_pub = nmrc.store_query
    
    # Make ROS messages
    nmrc.publish_ros_info(img_to_pub, nmrc.FRAME_ID, trueInd, matchInd, dvc, nmrc.ref_info['paths'][matchInd])

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