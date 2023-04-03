#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
import argparse as ap
import os
import sys

from scipy.spatial.distance import cdist

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom msg structures
from aarapsi_intro_pack.vpr_simple import VPRImageProcessor, Tolerance_Mode, FeatureType, labelImage, makeImage, grey2dToColourMap, \
                                            doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig
from aarapsi_intro_pack.core import enum_value_options, enum_get, enum_name, \
                                    check_positive_float, check_positive_two_int_tuple, check_positive_int, \
                                    check_bool, check_str_list, check_enum, check_string, \
                                    ROS_Param

class mrc: # main ROS class
    def __init__(self, database_path, ref_images_path, ref_odometry_path, image_feed_input, odometry_input, dataset_name, \
                    compress_in=True, compress_out=False, do_plotting=False, do_image=False, do_groundtruth=False, do_label=True, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), icon_settings=(50,20), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    match_metric='euclidean', namespace="/vpr_nodes", \
                    time_history_length=20, frame_id="base_link", \
                    node_name='vpr_all_in_one', anon=True, log_level=2, reset=False\
                ):
        
        self.NAMESPACE              = namespace
        self.NODENAME               = node_name
        self.NODESPACE              = "/" + self.NODENAME + "/"

        rospy.init_node(self.NODENAME, anonymous=anon, log_level=log_level)
        rospy.loginfo('Starting %s node.' % (node_name))

        ## Parse all the inputs:
        self.RATE_NUM               = ROS_Param(self.NODESPACE + "rate", rate_num, check_positive_float, force=reset) # Hz
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.FEAT_TYPE              = ROS_Param("feature_type", enum_name(ft_type), lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]), namespace=self.NAMESPACE, force=reset)
        self.IMG_DIMS               = ROS_Param("img_dims", img_dims, check_positive_two_int_tuple, namespace=self.NAMESPACE, force=reset)

        self.DATABASE_PATH          = ROS_Param("database_path", database_path, check_string, namespace=self.NAMESPACE, force=reset)
        self.REF_DATA_NAME          = ROS_Param(self.NODESPACE + "ref/data_name", dataset_name, check_string, force=reset)
        self.REF_IMG_PATH           = ROS_Param(self.NODESPACE + "ref/images_path", ref_images_path, check_string, force=reset)
        self.REF_ODOM_PATH          = ROS_Param(self.NODESPACE + "ref/odometry_path", ref_odometry_path, check_string, force=reset)

        self.FEED_TOPIC             = image_feed_input
        self.ODOM_TOPIC             = odometry_input

        self.TOL_MODE               = ROS_Param("tolerance/mode", enum_name(tolerance_mode), lambda x: check_enum(x, Tolerance_Mode), namespace=self.NAMESPACE, force=reset)
        self.TOL_THRES              = ROS_Param("tolerance/threshold", tolerance_threshold, check_positive_float, namespace=self.NAMESPACE, force=reset)

        self.ICON_SIZE              = icon_settings[0]
        self.ICON_DIST              = icon_settings[1]
        self.ICON_PATH              = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"

        self.MATCH_METRIC           = ROS_Param("match_metric", match_metric, check_string, namespace=self.NAMESPACE)
        self.TIME_HIST_LEN          = ROS_Param(self.NODESPACE + "time_history_length", time_history_length, check_positive_int, force=reset)
        self.FRAME_ID               = ROS_Param("frame_id", frame_id, check_string, namespace=self.NAMESPACE, force=reset)

        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN            = ROS_Param(self.NODESPACE + "compress/in", compress_in, check_bool, force=reset)
        self.COMPRESS_OUT           = ROS_Param(self.NODESPACE + "compress/out", compress_out, check_bool, force=reset)
        self.DO_PLOTTING            = ROS_Param(self.NODESPACE + "method/plotting", do_plotting, check_bool, force=reset)
        self.MAKE_IMAGE             = ROS_Param(self.NODESPACE + "method/images", do_image, check_bool)
        self.GROUND_TRUTH           = ROS_Param(self.NODESPACE + "method/groundtruth", do_groundtruth, check_bool, force=reset)
        self.MAKE_LABEL             = ROS_Param(self.NODESPACE + "method/label", do_label, check_bool, force=reset)

        self.ego                    = [0.0, 0.0, 0.0] # robot position

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor(ros=True, init_hybridnet=True, init_netvlad=True, cuda=True, dims=self.IMG_DIMS.get())
        try:
            self.ref_dict           = self.image_processor.npzDatabaseLoadSave(self.DATABASE_PATH.get(), self.REF_DATA_NAME.get(), \
                                                                                self.REF_IMG_PATH.get(), self.REF_ODOM_PATH.get(), \
                                                                                self.FEAT_TYPE.get(), self.IMG_DIMS.get(), do_save=False)
        except:
            self.exit()

        self.img_folder             = 'forward'

        self._compress_on           = {'topic': "/compressed", 'image': CompressedImage, 'label': CompressedImageLabelStamped}
        self._compress_off          = {'topic': "", 'image': Image, 'label': ImageLabelStamped}
        # Handle ROS details for input topics:
        if self.COMPRESS_IN.get():
            self.INPUTS             = self._compress_on
        else:
            self.INPUTS             = self._compress_off
        # Handle ROS details for output topics:
        if self.COMPRESS_OUT.get():
            self.OUTPUTS            = self._compress_on
        else:
            self.OUTPUTS            = self._compress_off
        
        self.param_checker_sub      = rospy.Subscriber("/vpr_nodes/params_update", String, self.param_callback, queue_size=100)

        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub       = rospy.Publisher(self.NAMESPACE + "/image" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)

        if self.MAKE_LABEL.get():
            self.img_sub            = rospy.Subscriber(self.FEED_TOPIC + self.INPUTS['topic'], self.INPUTS['image'], self.img_callback, queue_size=1) 
            self.odom_sub           = rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
            self.vpr_label_pub      = rospy.Publisher(self.NAMESPACE + "/label" + self.OUTPUTS['topic'], self.OUTPUTS['label'], queue_size=1)
            self.rolling_mtrx       = rospy.Publisher(self.NAMESPACE + "/matrices/rolling" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
            self.rolling_mtrx_img   = np.zeros((len(self.ref_dict['odom']['position']['x']), len(self.ref_dict['odom']['position']['x']))) # Make similarity matrix figure
        else:
            self.vpr_label_sub      = rospy.Subscriber(self.NAMESPACE + "/label" + self.INPUTS['topic'], self.INPUTS['label'], self.label_callback, queue_size=1)

        if self.DO_PLOTTING.get():
            self.fig, self.axes     = plt.subplots(1, 3, figsize=(15,4))
            self.timer_plot         = rospy.Timer(rospy.Duration(0.1), self.timer_plot_callback) # 10 Hz; Plot rate limiter

        # flags to denest main loop:
        self.new_query              = False # new query image (MAKE_LABEL.get()==True) or new label received (MAKE_LABEL.get()==False)
        self.new_odom               = False # new odometry set
        self.main_ready             = False # rate limiter via timer
        self.do_show                = False # plot rate limiter  via timer

        self.last_time              = rospy.Time.now()
        self.time_history           = []

        if self.DO_PLOTTING.get():
            # Prepare figures:
            self.fig.suptitle("Odometry Visualised")
            self.fig_mtrx_handles = doMtrxFig(self.axes[0], self.ref_dict['odom']) # Make simularity matrix figure
            self.fig_dvec_handles = doDVecFig(self.axes[1], self.ref_dict['odom']) # Make distance vector figure
            self.fig_odom_handles = doOdomFig(self.axes[2], self.ref_dict['odom']) # Make odometry figure
            self.fig.show()

        self.ICON_DICT = {'size': self.ICON_SIZE, 'dist': self.ICON_DIST, 'icon': [], 'good': [], 'poor': []}
        # Load icons:
        if self.MAKE_IMAGE.get():
            good_icon = cv2.imread(self.ICON_PATH + "/tick.png", cv2.IMREAD_UNCHANGED)
            poor_icon = cv2.imread(self.ICON_PATH + "/cross.png", cv2.IMREAD_UNCHANGED)
            self.ICON_DICT['good'] = cv2.resize(good_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)
            self.ICON_DICT['poor'] = cv2.resize(poor_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)

        # Last item as it sets a flag that enables main loop execution.
        self.main_timer             = rospy.Timer(rospy.Duration(1/self.RATE_NUM.get()), self.main_cb) # Main loop rate limiter

    def param_callback(self, msg):
        rospy.logdebug("Param update: %s" % msg.data)
        if (msg.data in self.RATE_NUM.updates_possible) and not (msg.data in self.RATE_NUM.updates_queued):
            self.RATE_NUM.updates_queued.append(msg.data)

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

        if self.COMPRESS_IN.get():
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

        if self.COMPRESS_IN.get():
            self.store_query_raw    = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            self.store_query_raw    = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.store_query            = self.store_query_raw#[10:-1,200:-50]
        self.new_query              = True
        
    def getMatchInd(self, ft_qry, metric='euclidean'):
    # top matching reference index for query

        dMat = cdist(self.ref_dict['img_feats'][enum_name(self.FEAT_TYPE.get())][self.img_folder], np.matrix(ft_qry), metric) # metric: 'euclidean' or 'cosine'
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

        self.rolling_mtrx_img = np.delete(self.rolling_mtrx_img, 0, 1) # delete first column (oldest query)
        self.rolling_mtrx_img = np.concatenate((self.rolling_mtrx_img, dvc), 1)
        
        mtrx_rgb = grey2dToColourMap(self.rolling_mtrx_img, dims=(500,500), colourmap=cv2.COLORMAP_JET)

        if self.COMPRESS_OUT.get():
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpeg") # jpeg (png slower)
            ros_matrix_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(mtrx_rgb, "jpeg") # jpeg (png slower)
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "bgr8")
            ros_matrix_to_pub = nmrc.bridge.cv2_to_imgmsg(mtrx_rgb, "bgr8")
        struct_to_pub = self.OUTPUTS['label']()
            
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
        struct_to_pub.data.compressed = self.COMPRESS_OUT.get()
        struct_to_pub.data.matchPath = mPath
        struct_to_pub.header.frame_id = fid
        struct_to_pub.header.stamp = rospy.Time.now()

        self.rolling_mtrx.publish(ros_matrix_to_pub)
        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub.publish(ros_image_to_pub) # image feed publisher
        if self.MAKE_LABEL.get():
            self.vpr_label_pub.publish(struct_to_pub) # label publisher

    def exit(self):
        rospy.loginfo("Quit received.")
        sys.exit()

def main_loop(nmrc):
# Main loop process

    if nmrc.do_show and nmrc.DO_PLOTTING.get(): # set by timer callback and node input
        nmrc.fig.canvas.draw() # update all fig subplots
        plt.pause(0.001)
        nmrc.do_show = False # clear flag

    if not (nmrc.new_query and (nmrc.new_odom or not nmrc.MAKE_LABEL.get()) and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    if (not nmrc.MAKE_LABEL.get()): # use label subscriber feed instead
        dvc             = np.transpose(np.matrix(nmrc.request.data.dvc))
        matchInd        = nmrc.request.data.matchId
        trueInd         = nmrc.request.data.trueId
    else:
        ft_qry          = nmrc.image_processor.getFeat(nmrc.store_query, nmrc.FEAT_TYPE.get(), use_tqdm=False)
        matchInd, dvc   = nmrc.getMatchInd(ft_qry, nmrc.MATCH_METRIC.get()) # Find match
        trueInd         = -1 #default; can't be negative.

    # Clear flags:
    nmrc.new_query      = False
    nmrc.new_odom       = False
    nmrc.main_ready     = False

    if nmrc.GROUND_TRUTH.get():
        trueInd = nmrc.getTrueInd() # find correct match based on shortest difference to measured odometry
    else:
        nmrc.ICON_DICT['size'] = -1

    ground_truth_string = ""
    tolState = 0
    if nmrc.GROUND_TRUTH.get(): # set by node inputs
        # Determine if we are within tolerance:
        nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['poor']
        if nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ego[1])) 
            tolString = "MCT"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][matchInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][matchInd] - nmrc.ego[1])) 
            tolString = "MCM"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ref_dict['odom']['position']['x'][matchInd]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ref_dict['odom']['position']['y'][matchInd])) 
            tolString = "ML"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
            tolString = "F"
        else:
            raise Exception("Error: Unknown tolerance mode.")

        if tolError < nmrc.TOL_THRES.get():
            nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['good']
            tolState = 2
        else:
            tolState = 1

        ground_truth_string = ", Error: %2.2f%s" % (tolError, tolString)

    if nmrc.MAKE_IMAGE.get(): # set by node input
        # make labelled match+query (processed) images and add icon for groundtruthing (if enabled):
        ft_ref = nmrc.ref_dict['img_feats'][enum_name(nmrc.FEAT_TYPE.get())][nmrc.img_folder][matchInd]
        if nmrc.FEAT_TYPE.get() in [FeatureType.NETVLAD, FeatureType.HYBRIDNET]:
            reshape_dims = (64, 64)
        else:
            reshape_dims = nmrc.IMG_DIMS.get()
        cv2_image_to_pub = makeImage(ft_qry, ft_ref, reshape_dims, nmrc.ICON_DICT)
        
        # Measure timing for recalculating average rate:
        this_time = rospy.Time.now()
        time_diff = this_time - nmrc.last_time
        nmrc.last_time = this_time
        nmrc.time_history.append(time_diff.to_sec())
        num_time = len(nmrc.time_history)
        if num_time > nmrc.TIME_HIST_LEN.get():
            nmrc.time_history.pop(0)
        time_average = sum(nmrc.time_history) / num_time

        # add label with feed information to image:
        label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + ground_truth_string
        cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

        img_to_pub = cv2_img_lab
    else:
        img_to_pub = nmrc.store_query

    if nmrc.DO_PLOTTING.get(): # set by node input
        # Update odometry visualisation:
        updateMtrxFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_mtrx_handles)
        updateDVecFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_dvec_handles)
        updateOdomFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_odom_handles)
    
    # Make ROS messages
    nmrc.publish_ros_info(img_to_pub, nmrc.FRAME_ID.get(), trueInd, matchInd, dvc, str(nmrc.ref_dict['image_paths']).replace('\'',''), tolState)

def do_args():
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
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset', '-R', type=check_bool, default=False, help='Force reset of parameters to specified ones (default: %(default)s)')

    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        
        # Hand to class ...
        nmrc = mrc(args['database-path'], args['ref-imgs-path'], args['ref-odom-path'], args['image-topic-in'], args['odometry-topic-in'], \
                    args['dataset-name'], compress_in=args['compress_in'], compress_out=args['compress_out'], do_plotting=args['do_plotting'], do_image=args['make_images'], \
                    do_groundtruth=args['groundtruth'], do_label=args['make_labels'], rate_num=args['rate'], ft_type=enum_get(args['ft_type'], FeatureType), \
                    img_dims=args['img_dims'], icon_settings=args['(size, distance)'], tolerance_threshold=args['tol_thresh'], \
                    tolerance_mode=enum_get(args['tol_mode'], Tolerance_Mode), match_metric='euclidean', namespace=args['namespace'], \
                    time_history_length=args['time_history_length'], frame_id=args['frame_id'], \
                    node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset']\
                )

        rospy.loginfo("Initialisation complete. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass

        