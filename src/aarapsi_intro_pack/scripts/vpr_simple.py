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

class Tolerance_Mode(Enum):
    METRE_CROW_TRUE = 0
    METRE_CROW_MATCH = 1
    METRE_LINE = 2
    FRAME = 3

class mrc: # main ROS class
    def __init__(self):

        rospy.init_node('vpr_simple', anonymous=True)
        rospy.loginfo('Starting vpr_simple node.')

        self.rate_num        = 20.0 # Hz, maximum of between 21-26 Hz (varies) with no plotting/image/ground truth/compression.
        self.rate_obj        = rospy.Rate(self.rate_num)

        #!# Tune Here:
        self.FEAT_TYPE       = "downsampled_raw" # Feature Type
        self.PACKAGE_NAME    = 'aarapsi_intro_pack'
        self.REF_IMG_PATH    = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/ccw_loop/forward" # Path where reference images are stored (names are sorted before reading)
        self.REF_ODOM_PATH   = rospkg.RosPack().get_path(self.PACKAGE_NAME) + "/data/ccw_loop/odo" # Path for where odometry .csv files are stored
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

        #!# Enable/Disable Features (Label topic will always be generated):
        self.DO_COMPRESS     = False
        self.DO_PLOTTING     = False
        self.MAKE_IMAGE      = True
        self.GROUND_TRUTH    = True

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
        self.new_img_frwd       = False
        self.new_odom           = False
        self.do_show            = False
        self.main_ready         = False

        self.last_time          = rospy.Time.now()
        self.time_history       = []

        # Process reference data (only needs to be done once)
        self.processRefOdomDataset(self.REF_ODOM_PATH)
        self.processRefImageDataset(self.REF_IMG_PATH, self.FEAT_TYPE, self.IMG_DIMS)

        if self.DO_PLOTTING:
            # Prepare figures:
            self.fig.suptitle("Odometry Visualised")
            doMtrxFig(self, axis=0) # Make simularity matrix figure
            doDVecFig(self, axis=1) # Make distance vector figure
            doOdomFig(self, axis=2) # Make odometry figure
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

        self.store_img_frwd     = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_img_frwd       = True

    def patchNormaliseImage(self, img, patchLength):
    # TODO: vectorize
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

    def getFeat(self, im, ftType, imDims):
        im = cv2.resize(im, imDims)
        ft = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if ftType == "downsampled_raw":
            pass # already done
        elif ftType == "downsampled_patchNorm":
            ft = self.patchNormaliseImage(ft, 8)
        return ft.flatten()

    def processRefImageDataset(self, path, ftType, imDims): 
    # Extract images and their features from path
    # Store in arrays and return them.

        imPath_list = np.sort(os.listdir(path))
        imPath_list = [os.path.join(path, f) for f in imPath_list]

        self.ref_info = {'paths': imPath_list, 'fts': []}

        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame, ftType, imDims) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
                self.ref_info['fts'].append(feat)
        else:
            rospy.logerr("Error: No files at reference image path - cannot continue.")
            raise Exception()
    
    def processRefOdomDataset(self, path):
    # Extract from position .csvs at path, robot x,y,z
    # Return these as nice lists

        self.ref_odom = {'x': [], 'y': [], 'z': []}
        odomPath_list = np.sort(os.listdir(path))
        odomPath_list = [os.path.join(path, f) for f in odomPath_list]
        if len(odomPath_list) > 0:
            for i, odomPath in tqdm(enumerate(odomPath_list)):
                new_odom = np.loadtxt(odomPath, delimiter=',')
                self.ref_odom['x'].append(new_odom[0])
                self.ref_odom['y'].append(new_odom[1])
                self.ref_odom['z'].append(new_odom[2])
        else:
            rospy.logerr("Error: No files at reference odometry path - cannot continue.")
            raise Exception()
        
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

    def publish_ros_info(self, cv2_img, fid, mInd, img_path):
        if self.DO_COMPRESS:
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpg") # jpg (png slower)
            struct_to_pub = CompressedImageLabelStamped()
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "passthrough")
            struct_to_pub = ImageLabelStamped()
            
        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = fid

        struct_to_pub.data.queryImage = ros_image_to_pub
        struct_to_pub.data.matchId = mInd
        struct_to_pub.data.matchPath = img_path
        struct_to_pub.header.frame_id = fid
        struct_to_pub.header.stamp = rospy.Time.now()

        if self.MAKE_IMAGE:
            self.vpr_feed_pub.publish(ros_image_to_pub)
        self.vpr_label_pub.publish(struct_to_pub)

##################################################################
#### Similarity Matrix Figure: do and update

def doMtrxFig(nmrc, axis):
    plt.sca(nmrc.axes[axis])
    mtrx_image = np.zeros((len(nmrc.ref_odom['x']), len(nmrc.ref_odom['x'])))
    mtrx_handle = nmrc.axes[axis].imshow(mtrx_image)
    nmrc.axes[axis].set(xlabel='Query Frame', ylabel='Reference Frame')

    nmrc.fig_mtrx_handles = {'img': mtrx_image, 'handle': mtrx_handle}

def updateMtrxFig(nmrc, mInd, tInd, dvc):
    img_new = np.delete(nmrc.fig_mtrx_handles['img'], 0, 1) # delete first column (oldest query)
    nmrc.fig_mtrx_handles['img'] = np.concatenate((img_new, np.array(dvc)), 1)
    nmrc.fig_mtrx_handles['handle'].set_data(nmrc.fig_mtrx_handles['img'])
    nmrc.fig_mtrx_handles['handle'].autoscale() # https://stackoverflow.com/questions/10970492/matplotlib-no-effect-of-set-data-in-imshow-for-the-plot

##################################################################
#### Distance Vector Figure: do and update

def doDVecFig(nmrc, axis):
# Set up distance vector figure
# https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.sca(nmrc.axes[axis]) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=7)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'mo', markersize=7)[0] # true image (correct match)
    nmrc.axes[axis].set(xlabel='Index', ylabel='Distance')
    nmrc.axes[axis].legend(["Image Distances", "Selected", "True"])
    nmrc.axes[axis].set_xlim(0, len(nmrc.ref_odom['x']))
    nmrc.axes[axis].set_ylim(0, 1.2)

    nmrc.fig_dvec_handles = {'axis': axis, 'dis': dist_vector, 'low': lowest_dist, 'act': actual_dist}

def updateDVecFig(nmrc, mInd, tInd, dvc):
# Update DVec figure with new data (match->mInd, true->tInd)
# update (overwrite) visualisation with new data:

    # overwrite with new distance vector / image distance:
    max_val = max(dvc[:])
    nmrc.fig_dvec_handles['dis'].set_xdata(range(len(dvc-1)))
    nmrc.fig_dvec_handles['dis'].set_ydata(dvc/max_val)
    # overwrite with new lowest match:
    nmrc.fig_dvec_handles['low'].set_xdata(mInd)
    nmrc.fig_dvec_handles['low'].set_ydata(dvc[mInd]/max_val)
    # overwrite with new truth value:
    nmrc.fig_dvec_handles['act'].set_xdata(tInd)
    nmrc.fig_dvec_handles['act'].set_ydata(dvc[tInd]/max_val)

##################################################################
#### Odometry Figure: do and update

def doOdomFig(nmrc, axis):
# Set up odometry figure

    plt.sca(nmrc.axes[axis])
    ref_plotted = plt.plot(nmrc.ref_odom['x'], nmrc.ref_odom['y'], 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    nmrc.axes[axis].set(xlabel='X-Axis', ylabel='Y-Axis')
    nmrc.axes[axis].legend(["Reference", "Match", "True"])
    nmrc.axes[axis].set_aspect('equal')

    nmrc.fig_odom_handles = {'axis': axis, 'ref': ref_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFig(nmrc, mInd, tInd):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Only display last 'queries_keep' number of points
    num_queries = len(list(nmrc.fig_odom_handles['tru'].get_xdata()))
    queries_keep = 10
    start_ind = num_queries - queries_keep + 1
    if num_queries < queries_keep:
        start_ind = 0
        
    ## odometry plot:
    # Append new value for "match" (what it matched the image to)
    nmrc.fig_odom_handles['mat'].set_xdata(np.append(nmrc.fig_odom_handles['mat'].get_xdata()[start_ind:num_queries], nmrc.ref_odom['x'][mInd]))
    nmrc.fig_odom_handles['mat'].set_ydata(np.append(nmrc.fig_odom_handles['mat'].get_ydata()[start_ind:num_queries], nmrc.ref_odom['y'][mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    nmrc.fig_odom_handles['tru'].set_xdata(np.append(nmrc.fig_odom_handles['tru'].get_xdata()[start_ind:num_queries], nmrc.ref_odom['x'][tInd]))
    nmrc.fig_odom_handles['tru'].set_ydata(np.append(nmrc.fig_odom_handles['tru'].get_ydata()[start_ind:num_queries], nmrc.ref_odom['y'][tInd]))

##################################################################
#### Image Tools:

def labelImage(img_in, textstring, org_in, colour):
# Write textstring at position org_in, with colour and black border on img_in

    # Black border:
    img_A = cv2.putText(img_in, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=(0,0,0), thickness=7, lineType=cv2.LINE_AA)
    # Colour inside:
    img_B = cv2.putText(img_A, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=colour, thickness=2, lineType=cv2.LINE_AA)
    return img_B

def makeImage(query_raw, match_path, icon_to_use, icon_size=100, icon_dist=0):
# Produce image to be published via ROS

    match_img = cv2.imread(match_path)
    query_img = cv2.resize(query_raw, (match_img.shape[1], match_img.shape[0]), interpolation = cv2.INTER_AREA) # resize to match_img dimensions
    
    match_img_lab = labelImage(match_img, "Reference", (20,40), (100,255,100))
    query_img_lab = labelImage(query_img, "Query", (20,40), (100,255,100))

    if icon_size > 0:
        # Add Icon:
        img_slice = match_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :]
        # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        icon_mask_inv = cv2.inRange(icon_to_use, (50,50,50), (255,255,255)) # get border (white)
        icon_mask = 255 - icon_mask_inv # get shape
        icon_mask_stack_inv = cv2.merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255 # stack into rgb layers, binary image
        icon_mask_stack = cv2.merge([icon_mask, icon_mask, icon_mask]) / 255 # stack into rgb layers, binary image
        opacity_icon = 0.8 # 80%
        # create new slice with appropriate layering
        img_slice = (icon_mask_stack_inv * img_slice) + \
                    (icon_mask_stack * icon_to_use) * (opacity_icon) + \
                    (icon_mask_stack * img_slice) * (1-opacity_icon)
        match_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :] = img_slice

    return np.concatenate((match_img_lab, query_img_lab), axis=1)

##################################################################
############################ MAIN ################################
##################################################################

def main_loop(nmrc):
# Main loop process

    if nmrc.do_show and nmrc.DO_PLOTTING:
        nmrc.fig.canvas.draw()
        nmrc.do_show = False

    if not (nmrc.new_img_frwd and nmrc.new_odom and nmrc.main_ready): # denest
        #rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    # Clear flags:
    nmrc.new_img_frwd   = False
    nmrc.new_odom       = False
    nmrc.main_ready     = False

    ft_qry = nmrc.getFeat(nmrc.store_img_frwd, nmrc.FEAT_TYPE, nmrc.IMG_DIMS).reshape(1,-1) # ensure 2D matrix
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
        updateDVecFig(nmrc, matchInd, trueInd, dvc)
        updateOdomFig(nmrc, matchInd, trueInd)
        updateMtrxFig(nmrc, matchInd, trueInd, dvc)

    if nmrc.MAKE_IMAGE:
        cv2_image_to_pub = makeImage(nmrc.store_img_frwd, nmrc.ref_info['paths'][matchInd], \
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

        # Make ROS messages
        img_to_pub = cv2_img_lab
    
    else:
        img_to_pub = nmrc.store_img_frwd
    
    nmrc.publish_ros_info(img_to_pub, nmrc.FRAME_ID, matchInd, nmrc.ref_info['paths'][matchInd])

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