#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
import argparse as ap
import os

from scipy.spatial.distance import cdist

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom msg structures
from aarapsi_intro_pack import VPRImageProcessor, Tolerance_Mode, FeatureType, labelImage, makeImage, \
                                doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get
from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool, check_str_list

class mrc: # main ROS class
    def __init__(self, database_path, ref_images_path, ref_odometry_path, image_feed_input, odometry_input, dataset_name, \
                    compress_in=True, compress_out=False, do_plotting=False, do_image=False, do_groundtruth=False, do_label=True, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), icon_settings=(50,20), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    match_metric='euclidean', namespace="/vpr_nodes", \
                    time_history_length=20, frame_id="base_link", \
                    node_name='vpr_all_in_one', anon=True\
                ):

        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        ## Parse all the inputs:
        self.rate_num               = rate_num # Hz
        self.rate_obj               = rospy.Rate(self.rate_num)

        self.FEAT_TYPE              = ft_type
        self.IMG_DIMS               = img_dims

        self.DATABASE_PATH          = database_path
        self.REF_DATA_NAME          = dataset_name
        self.REF_IMG_PATH           = ref_images_path
        self.REF_ODOM_PATH          = ref_odometry_path

        self.NAMESPACE              = namespace
        self.FEED_TOPIC             = image_feed_input
        self.ODOM_TOPIC             = odometry_input

        self.TOL_MODE               = tolerance_mode
        self.TOL_THRES              = tolerance_threshold

        self.ICON_SIZE              = icon_settings[0]
        self.ICON_DIST              = icon_settings[1]
        self.ICON_PATH              = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"

        self.MATCH_METRIC           = match_metric
        self.TIME_HIST_LEN          = time_history_length
        self.FRAME_ID               = frame_id

        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN            = compress_in
        self.COMPRESS_OUT           = compress_out
        self.DO_PLOTTING            = do_plotting
        self.MAKE_IMAGE             = do_image
        self.GROUND_TRUTH           = do_groundtruth
        self.MAKE_LABEL             = do_label

        self.ego                    = [0.0, 0.0, 0.0] # robot position

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        # Handle ROS details for input topics:
        if self.COMPRESS_IN:
            self.in_img_tpc_mode    = "/compressed"
            self.in_image_type      = CompressedImage
            self.in_label_type      = CompressedImageLabelStamped
        else:
            self.in_img_tpc_mode    = ""
            self.in_image_type      = Image
            self.in_label_type      = ImageLabelStamped
        # Handle ROS details for output topics:
        if self.COMPRESS_OUT:
            self.out_img_tpc_mode   = "/compressed"
            self.out_image_type     = CompressedImage
            self.out_label_type     = CompressedImageLabelStamped
        else:
            self.out_img_tpc_mode   = ""
            self.out_image_type     = Image
            self.out_label_type     = ImageLabelStamped
        
        if self.MAKE_IMAGE:
            self.vpr_feed_pub       = rospy.Publisher(self.NAMESPACE + "/image" + self.out_img_tpc_mode, self.out_image_type, queue_size=1)

        if self.MAKE_LABEL:
            self.img_sub            = rospy.Subscriber(self.FEED_TOPIC + self.in_img_tpc_mode, self.in_image_type, self.img_callback, queue_size=1) 
            self.odom_sub           = rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
            self.vpr_label_pub      = rospy.Publisher(self.NAMESPACE + "/label" + self.out_img_tpc_mode, self.out_label_type, queue_size=1)
        else:
            self.vpr_label_sub      = rospy.Subscriber(self.NAMESPACE + "/label" + self.in_img_tpc_mode, self.in_label_type, self.label_callback, queue_size=1)

        if self.DO_PLOTTING:
            self.fig, self.axes     = plt.subplots(1, 3, figsize=(15,4))
            self.timer_plot         = rospy.Timer(rospy.Duration(0.1), self.timer_plot_callback) # 10 Hz; Plot rate limiter

        # flags to denest main loop:
        self.new_query              = False # new query image (MAKE_LABEL==True) or new label received (MAKE_LABEL==False)
        self.new_odom               = False # new odometry set
        self.main_ready             = False # rate limiter via timer
        self.do_show                = False # plot rate limiter  via timer

        self.last_time              = rospy.Time.now()
        self.time_history           = []

        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor()
        self.ref_dict               = self.image_processor.npzDatabaseLoadSave(self.DATABASE_PATH, self.REF_DATA_NAME, \
                                                                                self.REF_IMG_PATH, self.REF_ODOM_PATH, \
                                                                                self.FEAT_TYPE, self.IMG_DIMS, do_save=True)
        self.img_folder             = 'forward'

        if self.DO_PLOTTING:
            # Prepare figures:
            self.fig.suptitle("Odometry Visualised")
            self.fig_mtrx_handles = doMtrxFig(self.axes[0], self.ref_dict['odom']) # Make simularity matrix figure
            self.fig_dvec_handles = doDVecFig(self.axes[1], self.ref_dict['odom']) # Make distance vector figure
            self.fig_odom_handles = doOdomFig(self.axes[2], self.ref_dict['odom']) # Make odometry figure
            self.fig.show()

        self.ICON_DICT = {'size': self.ICON_SIZE, 'dist': self.ICON_DIST, 'icon': [], 'good': [], 'poor': []}
        # Load icons:
        if self.MAKE_IMAGE:
            good_icon = cv2.imread(self.ICON_PATH + "/tick.png", cv2.IMREAD_UNCHANGED)
            poor_icon = cv2.imread(self.ICON_PATH + "/cross.png", cv2.IMREAD_UNCHANGED)
            self.ICON_DICT['good'] = cv2.resize(good_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)
            self.ICON_DICT['poor'] = cv2.resize(poor_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)

        # Last item as it sets a flag that enables main loop execution.
        self.main_timer             = rospy.Timer(rospy.Duration(1/self.rate_num), self.main_cb) # Main loop rate limiter

    def main_cb(self, event):
    # Toggle flag to let main loop continue execution
    # This is currently bottlenecked by node performance and a rate limiter regardless, but I have kept it for future work

        self.main_ready = True

    def timer_plot_callback(self, event):
    # Toggle flag so that visualisation is performed at a lower rate than main loop processing (10Hz instead of >>10Hz)

        self.do_show            = True

    def label_callback(self, msg):
    # /vpr_nodes/label(/compressed) (aarapsi_intro_pack/(Compressed)ImageLabelStamped)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.request            = msg

        if self.request.data.trueId < 0:
            self.GROUND_TRUTH   = False

        self.ego                = [msg.data.odom.x, msg.data.odom.y, msg.data.odom.z]

        if self.COMPRESS_IN:
            self.store_query    = self.bridge.compressed_imgmsg_to_cv2(self.request.queryImage, "passthrough")
        else:
            self.store_query    = self.bridge.imgmsg_to_cv2(self.request.queryImage, "passthrough")

        self.new_query          = True

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.ego                = [round(msg.pose.pose.position.x, 3), round(msg.pose.pose.position.y, 3), round(msg.pose.pose.position.z, 3)]
        self.new_odom           = True

    def img_callback(self, msg):
    # /ros_indigosdk_occam/image0(/compressed) (sensor_msgs/(Compressed)Image)
    # Store newest image received

        if self.COMPRESS_IN:
            self.store_query_raw    = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            self.store_query_raw    = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.store_query = self.store_query_raw#[10:-1,200:-50]
        self.new_query          = True
        
    def getMatchInd(self, ft_qry, metric='euclidean'):
    # top matching reference index for query

        dMat = cdist(self.ref_dict['img_feats'][self.img_folder], ft_qry, metric) # metric: 'euclidean' or 'cosine'
        mInd = np.argmin(dMat[:])
        return mInd, dMat
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares = np.square(np.array(self.ref_dict['odom']['position']['x']) - self.ego[0]) + \
                            np.square(np.array(self.ref_dict['odom']['position']['y']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, cv2_img, fid, tInd, mInd, dvc, mPath, state):
    # Publish label and/or image feed
        if self.COMPRESS_OUT:
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpg") # jpg (png slower)
            struct_to_pub = CompressedImageLabelStamped()
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "passthrough")
            struct_to_pub = ImageLabelStamped()
            
        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = fid

        struct_to_pub.queryImage = ros_image_to_pub
        struct_to_pub.data.odom.x = self.ego[0]
        struct_to_pub.data.odom.y = self.ego[1]
        struct_to_pub.data.odom.z = self.ego[2]
        struct_to_pub.data.dvc = dvc
        struct_to_pub.data.matchId = mInd
        struct_to_pub.data.trueId = tInd
        struct_to_pub.data.state = state
        struct_to_pub.data.compressed = self.COMPRESS_OUT
        struct_to_pub.data.matchPath = mPath
        struct_to_pub.header.frame_id = fid
        struct_to_pub.header.stamp = rospy.Time.now()

        if self.MAKE_IMAGE:
            self.vpr_feed_pub.publish(ros_image_to_pub) # image feed publisher
        if self.MAKE_LABEL:
            self.vpr_label_pub.publish(struct_to_pub) # label publisher

def main_loop(nmrc):
# Main loop process

    if nmrc.do_show and nmrc.DO_PLOTTING: # set by timer callback and node input
        nmrc.fig.canvas.draw() # update all fig subplots
        nmrc.do_show = False # clear flag

    if not (nmrc.new_query and (nmrc.new_odom or not nmrc.MAKE_LABEL) and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    if (not nmrc.MAKE_LABEL): # use label subscriber feed instead
        dvc             = np.transpose(np.matrix(nmrc.request.data.dvc))
        matchInd        = nmrc.request.data.matchId
        trueInd         = nmrc.request.data.trueId
    else:
        ft_qry          = nmrc.image_processor.getFeat(nmrc.store_query, size=2)
        matchInd, dvc   = nmrc.getMatchInd(ft_qry, nmrc.MATCH_METRIC) # Find match
        trueInd         = -1 #default; can't be negative.

    # Clear flags:
    nmrc.new_query      = False
    nmrc.new_odom       = False
    nmrc.main_ready     = False

    if nmrc.GROUND_TRUTH:
        trueInd = nmrc.getTrueInd() # find correct match based on shortest difference to measured odometry
    else:
        nmrc.ICON_DICT['size'] = -1

    ground_truth_string = ""
    tolState = 0
    if nmrc.GROUND_TRUTH: # set by node inputs
        # Determine if we are within tolerance:
        nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['poor']
        if nmrc.TOL_MODE == Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ego[1])) 
            tolString = "MCT"
        elif nmrc.TOL_MODE == Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][matchInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][matchInd] - nmrc.ego[1])) 
            tolString = "MCM"
        elif nmrc.TOL_MODE == Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ref_dict['odom']['position']['x'][matchInd]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ref_dict['odom']['position']['y'][matchInd])) 
            tolString = "ML"
        elif nmrc.TOL_MODE == Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
            tolString = "F"
        else:
            raise Exception("Error: Unknown tolerance mode.")

        if tolError < nmrc.TOL_THRES:
            nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['good']
            tolState = 2
        else:
            tolState = 1

        ground_truth_string = ", Error: %2.2f%s" % (tolError, tolString)

    if nmrc.DO_PLOTTING: # set by node input
        # Update odometry visualisation:
        updateMtrxFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_mtrx_handles)
        updateDVecFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_dvec_handles)
        updateOdomFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_odom_handles)

    if nmrc.MAKE_IMAGE: # set by node input
        # make labelled match+query image and add icon for groundtruthing (if enabled):
        cv2_image_to_pub = makeImage(nmrc.store_query, np.reshape(np.array(nmrc.ref_dict['img_feats'][nmrc.img_folder])[matchInd,:], nmrc.ref_dict['img_dims']), \
                                        nmrc.ICON_DICT['icon'], icon_size=nmrc.ICON_DICT['size'], icon_dist=nmrc.ICON_DICT['dist'])
        
        # Measure timing for recalculating average rate:
        this_time = rospy.Time.now()
        time_diff = this_time - nmrc.last_time
        nmrc.last_time = this_time
        nmrc.time_history.append(time_diff.to_sec())
        num_time = len(nmrc.time_history)
        if num_time > nmrc.TIME_HIST_LEN:
            nmrc.time_history.pop(0)
        time_average = sum(nmrc.time_history) / num_time

        # add label with feed information to image:
        label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + ground_truth_string
        cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

        img_to_pub = cv2_img_lab
    else:
        img_to_pub = nmrc.store_query
    
    # Make ROS messages
    nmrc.publish_ros_info(img_to_pub, nmrc.FRAME_ID, trueInd, matchInd, dvc, str(nmrc.ref_dict['image_paths']).replace('\'',''), tolState)

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="vpr_all_in_one", 
                                   description="ROS implementation of QVPR's VPR Primer",
                                   epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        
        # Positional Arguments:
        parser.add_argument('dataset-name', help='Specify name of dataset (for fast loading; matches are made on names starting with provided string).')
        parser.add_argument('database-path', help="Specify path to where compressed databases exist (for fast loading).")
        parser.add_argument('ref-imgs-path', type=check_str_list, help="Specify path to reference images (for slow loading).")
        parser.add_argument('ref-odom-path', help="Specify path to reference odometry (for slow loading).")
        parser.add_argument('image-topic-in', help="Specify input image topic (exclude /compressed).")
        parser.add_argument('odometry-topic-in', help="Specify input odometry topic (exclude /compressed).")

        # Optional Arguments:
        parser.add_argument('--compress-in', '-Ci', type=check_bool, default=False, help='Enable image compression on input (default: %(default)s)')
        parser.add_argument('--compress-out', '-Co', type=check_bool, default=False, help='Enable image compression on output (default: %(default)s)')
        parser.add_argument('--do-plotting', '-P', type=check_bool, default=False, help='Enable matplotlib visualisations (default: %(default)s)')
        parser.add_argument('--make-images', '-I', type=check_bool, default=False, help='Enable image topic generation (default: %(default)s)')
        parser.add_argument('--groundtruth', '-G', type=check_bool, default=False, help='Enable groundtruth inclusion (default: %(default)s)')
        parser.add_argument('--make-labels', '-L', type=check_bool, default=True, help='Enable label topic generation; false enables a subscriber instead (default: %(default)s)')
        parser.add_argument('--rate', '-r', type=check_positive_float, default=10.0, help='Set node rate (default: %(default)s).')
        parser.add_argument('--time-history-length', '-l', type=check_positive_int, default=10, help='Set keep history size for logging true rate (default: %(default)s).')
        parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
        ft_options, ft_options_text = enum_value_options(FeatureType, skip=FeatureType.NONE)
        parser.add_argument('--ft-type', '-F', type=int, choices=ft_options, default=ft_options[0], \
                            help='Choose feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
        tolmode_options, tolmode_options_text = enum_value_options(Tolerance_Mode)
        parser.add_argument('--tol-mode', '-t', type=int, choices=tolmode_options, default=tolmode_options[0], \
                            help='Choose tolerance mode for ground truth, types: %s (default: %s).' % (tolmode_options_text, '%(default)s'))
        parser.add_argument('--tol-thresh', '-T', type=check_positive_float, default=1.0, help='Set tolerance threshold for ground truth (default: %(default)s).')
        parser.add_argument('--icon-info', '-p', dest='(size, distance)', type=check_positive_two_int_tuple, default=(50,20), help='Set icon (size, distance) (default: %(default)s).')
        parser.add_argument('--node-name', '-N', default="vpr_all_in_one", help="Specify node name (default: %(default)s).")
        parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--namespace', '-n', default="/vpr_nodes", help="Specify namespace for topics (default: %(default)s).")
        parser.add_argument('--frame-id', '-f', default="base_link", help="Specify frame_id for messages (default: %(default)s).")

        # Parse args...
        raw_args = parser.parse_known_args()
        args = vars(raw_args[0])
        

        # Hand to class ...
        nmrc = mrc(args['database-path'], args['ref-imgs-path'], args['ref-odom-path'], args['image-topic-in'], args['odometry-topic-in'], \
                    args['dataset-name'], compress_in=args['compress_in'], compress_out=args['compress_out'], do_plotting=args['do_plotting'], do_image=args['make_images'], \
                    do_groundtruth=args['groundtruth'], do_label=args['make_labels'], rate_num=args['rate'], ft_type=enum_get(args['ft_type'], FeatureType), \
                    img_dims=args['img_dims'], icon_settings=args['(size, distance)'], tolerance_threshold=args['tol_thresh'], \
                    tolerance_mode=enum_get(args['tol_mode'], Tolerance_Mode), match_metric='euclidean', namespace=args['namespace'], \
                    time_history_length=args['time_history_length'], frame_id=args['frame_id'], \
                    node_name=args['node_name'], anon=args['anon']\
                )

        rospy.loginfo("Initialisation complete. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass

        