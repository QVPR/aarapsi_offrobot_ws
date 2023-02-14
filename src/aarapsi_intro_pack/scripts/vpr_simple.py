#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
## https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from enum import Enum

import os
from scipy.spatial.distance import cdist
from tqdm import tqdm
from aarapsi_intro_pack.msg import CompressedImageLabelStamped # Our custom structure

class Tolerance_Mode(Enum):
    METRE = 0
    FRAME = 1

class Tolerance_State(Enum):
    GOOD = 0
    POOR = 1

class mrc: # main ROS class
    def __init__(self, sub_image_topic, sub_odom_topic):
        self.robot_x            = 0
        self.robot_y            = 0
        self.robot_z            = 0

        rospy.init_node('vpr_simple', anonymous=True)
        rospy.loginfo('Starting vpr_simple node.')

        self.fig, self.axes     = plt.subplots(1, 3)

        self.rate_num           = 50 # Hz
        self.rate_obj           = rospy.Rate(self.rate_num)

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        self.img_frwd_sub       = rospy.Subscriber(sub_image_topic, CompressedImage, self.img_frwd_callback, queue_size=1) # frwd-view 
        self.odom_sub           = rospy.Subscriber(sub_odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.vpr_feed_pub       = rospy.Publisher("/vpr_simple/image/compressed", CompressedImage, queue_size=1)
        self.vpr_label_pub      = rospy.Publisher("/vpr_simple/image/label", CompressedImageLabelStamped, queue_size=1)
        self.timer_plot         = rospy.Timer(rospy.Duration(0.1), self.timer_plot_callback) # 10 Hz

        # flags to denest main loop:
        self.new_img_frwd       = False
        self.new_odom           = False
        self.last_time          = rospy.Time.now()
        self.main_ready         = False

    def timer_plot_callback(self, event):
        if not self.main_ready:
            return
        
        #self.fig.show()
        plt.draw()
        #plt.pause(0.000001)

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

    def processImageDataset(self, path, ftType, imDims): 
    # Extract images and their features from path
    # Store in arrays and return them.

        imPath_list = np.sort(os.listdir(path))
        imPath_list = [os.path.join(path, f) for f in imPath_list]
        feat_list = []
        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame, ftType, imDims) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
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

def doMtrxFig(axes, ref_x_data, ref_y_data):
    plt.sca(axes)
    mtrx_image = np.zeros((len(ref_x_data), len(ref_x_data)))
    axes.imshow(mtrx_image)
    axes.set(xlabel='Query Frame', ylabel='Reference Frame')
    return mtrx_image

def doDVecFig(axes, ref_x_data, ref_y_data):
# Set up distance vector figure
# https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.sca(axes) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=7)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'mo', markersize=7)[0] # true image (correct match)
    axes.set(xlabel='Index', ylabel='Distance')
    axes.legend(["Image Distances", "Selected", "True"])
    axes.set_xlim(0, len(ref_x_data))
    axes.set_ylim(0, 10000)

    return dist_vector, lowest_dist, actual_dist # return handles

def doOdomFig(axes, ref_x_data, ref_y_data):
# Set up odometry figure

    plt.sca(axes)
    ref_plotted = plt.plot(ref_x_data, ref_y_data, 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    axes.set(xlabel='X-Axis', ylabel='Y-Axis')
    axes.legend(["Reference", "Match", "True"])

    stretch_x_ref_data = 0.1 * (max(ref_x_data) - min(ref_x_data))
    stretch_y_ref_data = 0.1 * (max(ref_y_data) - min(ref_y_data))

    axes.set_xlim(min(ref_x_data) - stretch_x_ref_data, max(ref_x_data) + stretch_x_ref_data)
    axes.set_ylim(min(ref_y_data) - stretch_y_ref_data, max(ref_y_data) + stretch_y_ref_data)
    axes.set_aspect('equal')

    return ref_plotted, mat_plotted, tru_plotted # return handles

def updateMtrxFig(matchInd, trueInd, mtrx, axes, new_dv):
    mtrx_new = np.delete(mtrx, 0, 1) # delete first column (oldest query)
    mtrx_new = np.concatenate((mtrx_new, np.array(new_dv)), 1)
    axes.clear()
    axes.imshow(mtrx_new, interpolation=None, alpha=None, origin='centre',extent=(0,len(new_dv),0,len(new_dv)))
    return mtrx_new

def updateDVecFig(mInd, tInd, dis, low, act, dvc):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (dist_vector, lowest_dist, actual_dist) and crunched distance vector (dvc)
    ## distance vector plot:
    # overwrite with new distance vector / image distance:
    dis.set_xdata(range(len(dvc-1)))
    dis.set_ydata(dvc)
    # overwrite with new lowest match:
    low.set_xdata(mInd)
    low.set_ydata(dvc[mInd])
    # overwrite with new truth value:
    act.set_xdata(tInd)
    act.set_ydata(dvc[tInd])

def updateOdomFig(mInd, tInd, ref, mat, tru, x_data, y_data):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Only display last 'queries_keep' number of points
    num_queries = len(list(tru.get_xdata()))
    queries_keep = 10
    start_ind = num_queries - queries_keep + 1
    if num_queries < queries_keep:
        start_ind = 0
        
    ## odometry plot:
    # Append new value for "match" (what it matched the image to)
    mat.set_xdata(np.append(mat.get_xdata()[start_ind:num_queries], x_data[mInd]))
    mat.set_ydata(np.append(mat.get_ydata()[start_ind:num_queries], y_data[mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    tru.set_xdata(np.append(tru.get_xdata()[start_ind:num_queries], x_data[tInd]))
    tru.set_ydata(np.append(tru.get_ydata()[start_ind:num_queries], y_data[tInd]))

def makeImage(query_raw, match_path, icon_to_use, icon_size=100, icon_dist=0):
    match_img = cv2.imread(match_path)
    query_img = cv2.resize(query_raw, (match_img.shape[1], match_img.shape[0]), interpolation = cv2.INTER_AREA) # resize to match_img dimensions
    
    match_img_lab = labelImage(match_img, "Reference", (20,40), (100,255,100))
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
    query_img_lab = labelImage(query_img, "Query", (20,40), (100,255,100))

    return np.concatenate((match_img_lab, query_img_lab), axis=1)

def vpr_simple():

    #!# Tune Here:
    ftType          = "downsampled_raw" # Feature Type
    refImagesPath   = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/forward" # Path where reference images are stored (names are sorted before reading)
    refOdomPath     = rospkg.RosPack().get_path('aarapsi_intro_pack') + "/data/cw_loop/odo" # Path for where odometry .csv files are stored
    feed_topic      = "/ros_indigosdk_occam/image0/compressed"
    odom_topic      = "/odometry/filtered"
    tolMode         = Tolerance_Mode.METRE # or FRAME
    tolThres        = 5.0
    FRAME_ID        = "base_link"
    ICON_SIZE       = 50
    ICON_DIST       = 20
    imDims          = (64, 64)
    match_metric    = 'euclidean'

    nmrc = mrc(feed_topic, odom_topic) # make new class instance

    # Load icons:
    good_icon = cv2.imread(rospkg.RosPack().get_path('aarapsi_intro_pack') + '/media/' + 'tick.png', cv2.IMREAD_UNCHANGED)
    poor_icon = cv2.imread(rospkg.RosPack().get_path('aarapsi_intro_pack') + '/media/' + 'cross.png', cv2.IMREAD_UNCHANGED)
    good_small_icon = cv2.resize(good_icon, (ICON_SIZE, ICON_SIZE), interpolation = cv2.INTER_AREA)
    poor_small_icon = cv2.resize(poor_icon, (ICON_SIZE, ICON_SIZE), interpolation = cv2.INTER_AREA)

    ## Prepare lists for plotting via plt
    # Reference: Full list for use in query and true
    # Match: the odom for the matched image
    # True: for n'th index, the correct odom from ROS topic
    ref_x_data, ref_y_data, _ = nmrc.processOdomDataset(refOdomPath)
    if len(ref_x_data) < 1:
       return # error case; list is empty!

    # Process reference data (only needs to be done once)
    imgList_ref_paths, ft_ref = nmrc.processImageDataset(refImagesPath, ftType, imDims)
    if len(imgList_ref_paths) < 1:
       return # error case; list is empty!

    nmrc.fig.suptitle("Odometry Visualised")
    mtrx_lists = doMtrxFig(nmrc.axes[0], ref_x_data, ref_y_data) # Make simularity matrix figure
    dvec_lists = doDVecFig(nmrc.axes[1], ref_x_data, ref_y_data) # Make distance vector figure
    odom_lists = doOdomFig(nmrc.axes[2], ref_x_data, ref_y_data) # Make odometry figure
    nmrc.fig.show()
    plt.pause(0.001)

    rospy.loginfo("Reference list processed. Listening for queries...")
    
    # Main loop:
    while not rospy.is_shutdown():
        nmrc.main_ready = True
        nmrc.rate_obj.sleep()

        if not (nmrc.new_img_frwd and nmrc.new_odom): # denest
           rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
           continue

        # Clear flags:
        nmrc.new_img_frwd   = False
        nmrc.new_odom       = False

        ft_qry = nmrc.getFeat(nmrc.store_img_frwd, ftType, imDims).reshape(1,-1) # ensure 2D matrix
        matchInd, dist_vector_calc = nmrc.getMatchInds(ft_ref, ft_qry, match_metric) # Find match/es
        trueInd = nmrc.getTrueInds((nmrc.robot_x, nmrc.robot_y), ref_x_data, ref_y_data) # find correct match based on shortest difference to measured odometry

        # Determine if we are within tolerance:
        tolState = Tolerance_State.POOR # assume poor
        icon_to_use = poor_small_icon
        if tolMode == Tolerance_Mode.METRE:
            tolError = np.sqrt(np.square(ref_x_data[trueInd] - ref_x_data[matchInd]) + \
                       np.square(ref_y_data[trueInd] - ref_y_data[matchInd])) 
            tolString = "M"
        elif tolMode == Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
            tolString = "F"

        if tolError < tolThres:
            tolState = Tolerance_State.GOOD
            icon_to_use = good_small_icon

        # Update odometry visualisation:
        updateDVecFig(matchInd, trueInd, dvec_lists[0], dvec_lists[1], dvec_lists[2], dist_vector_calc)
        updateOdomFig(matchInd, trueInd, odom_lists[0], odom_lists[1], odom_lists[2], ref_x_data, ref_y_data)
        mtrx_lists = updateMtrxFig(matchInd, trueInd, mtrx_lists, nmrc.axes[0], dist_vector_calc)

        try:
            cv2_image_to_pub = makeImage(nmrc.store_img_frwd, imgList_ref_paths[matchInd], \
                                         icon_to_use, icon_size=ICON_SIZE, icon_dist=ICON_DIST)
            
            # Measure timing
            this_time = rospy.Time.now()
            time_diff = this_time - nmrc.last_time
            nmrc.last_time = this_time

            label_string = "Index [%04d], %2.2f Hz, Error: %2.2f%s" % (matchInd, 1/time_diff.to_sec(), tolError, tolString)
            cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

            # Make ROS messages
            ros_image_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(cv2_img_lab, "png")
            ros_image_to_pub.header.stamp = rospy.Time.now()
            ros_image_to_pub.header.frame_id = FRAME_ID

            struct_to_pub = CompressedImageLabelStamped()
            struct_to_pub.data.queryImage = ros_image_to_pub
            struct_to_pub.data.matchId = matchInd
            struct_to_pub.data.matchPath = imgList_ref_paths[matchInd]
            struct_to_pub.header.frame_id = ros_image_to_pub.header.frame_id
            struct_to_pub.header.stamp = rospy.Time.now()

            nmrc.vpr_feed_pub.publish(ros_image_to_pub)
            nmrc.vpr_label_pub.publish(struct_to_pub)

        except Exception as e:
           rospy.logerr("Error caught in ROS msg processing. K condition violation likely (K = 0?). Caught error:")
           rospy.logwarn(e)
           return


if __name__ == '__main__':
    try:
        vpr_simple()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass