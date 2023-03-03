#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import numpy as np
import rospkg
from matplotlib import pyplot as plt
import argparse as ap
import os
import sys
from scipy.spatial.distance import cdist

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped # Our custom structures
from aarapsi_intro_pack import VPRImageProcessor, Tolerance_Mode, FeatureType
from aarapsi_intro_pack.vpred import *
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get
from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool

from sklearn import svm
from sklearn.preprocessing import StandardScaler

class mrc: # main ROS class
    def __init__(self, ref_dataset_name, cal_qry_dataset_name, cal_ref_dataset_name, database_path, image_feed_input, odometry_input, \
                    compress_in=True, compress_out=False, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), icon_settings=(50,20), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    match_metric='euclidean', namespace="/vpr_nodes", \
                    time_history_length=20, frame_id="base_link", \
                    node_name='vpr_monitor', anon=True,\
                    cal_qry_folder='forward',\
                    cal_ref_folder='forward',\
                    ref_ref_folder='forward'\
                ):

        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.rate_num           = rate_num # Hz, maximum of between 21-26 Hz (varies) with no plotting/image/ground truth/compression.
        self.rate_obj           = rospy.Rate(self.rate_num)

        #!# Tune Here:
        self.FEAT_TYPE          = ft_type
        self.IMG_DIMS           = img_dims

        self.DATABASE_PATH      = database_path
        self.REF_DATA_NAME      = ref_dataset_name
        self.CAL_QRY_DATA_NAME  = cal_qry_dataset_name
        self.CAL_REF_DATA_NAME  = cal_ref_dataset_name

        self.CAL_QRY_FOLDER     = cal_qry_folder
        self.CAL_REF_FOLDER     = cal_ref_folder
        self.REF_REF_FOLDER     = ref_ref_folder

        self.NAMESPACE          = namespace
        self.FEED_TOPIC         = image_feed_input
        self.ODOM_TOPIC         = odometry_input

        self.TOL_MODE           = tolerance_mode
        self.TOL_THRES          = tolerance_threshold

        self.ICON_SIZE          = icon_settings[0]
        self.ICON_DIST          = icon_settings[1]
        self.ICON_PATH          = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"

        self.MATCH_METRIC       = match_metric
        self.TIME_HIST_LEN      = time_history_length
        self.FRAME_ID           = frame_id

        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN        = compress_in
        self.COMPRESS_OUT       = compress_out

        self.ego                = [0.0, 0.0, 0.0] # robot position

        self.bridge             = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        if self.COMPRESS_IN:
            self.in_img_tpc_mode   = "/compressed"
            self.in_image_type     = CompressedImage
            self.in_label_type     = CompressedImageLabelStamped
        else:
            self.in_img_tpc_mode   = ""
            self.in_image_type     = Image
            self.in_label_type     = ImageLabelStamped

        if self.COMPRESS_OUT:
            self.out_img_tpc_mode   = "/compressed"
            self.out_image_type     = CompressedImage
            self.out_label_type     = CompressedImageLabelStamped
        else:
            self.out_img_tpc_mode   = ""
            self.out_image_type     = Image
            self.out_label_type     = ImageLabelStamped

        # Process reference data (only needs to be done once)
        rospy.logdebug("Loading reference data set...")
        self.ref_ip             = VPRImageProcessor()
        if not self.ref_ip.npzLoader(self.DATABASE_PATH, self.REF_DATA_NAME):
            self.exit()

        # Process calibration data (only needs to be done once)
        rospy.logdebug("Loading calibration query data set...")
        self.cal_qry_ip             = VPRImageProcessor()
        if not self.cal_qry_ip.npzLoader(self.DATABASE_PATH, self.CAL_QRY_DATA_NAME):
            self.exit()
        rospy.logdebug("Loading calibration reference data set...")
        self.cal_ref_ip             = VPRImageProcessor()
        if not self.cal_ref_ip.npzLoader(self.DATABASE_PATH, self.CAL_REF_DATA_NAME):
            self.exit()

        self.calibrate()
        self.train()

        self.vpr_label_sub      = rospy.Subscriber(self.NAMESPACE + "/label" + self.in_img_tpc_mode, self.in_label_type, self.label_callback, queue_size=1)

        # flags to denest main loop:
        self.new_query              = False # new label received
        self.main_ready             = False # rate limiter via timer

        self.last_time              = rospy.Time.now()
        self.time_history           = []

        # Last item as it sets a flag that enables main loop execution.
        self.main_timer             = rospy.Timer(rospy.Duration(1/self.rate_num), self.main_cb) # Main loop rate limiter

        self.states                 = [0,0,0]

    def main_cb(self, event):
    # Toggle flag to let main loop continue execution
    # This is currently bottlenecked by node performance and a rate limiter regardless, but I have kept it for future work

        self.main_ready = True

    def label_callback(self, msg):
    # /vpr_nodes/label(/compressed) (aarapsi_intro_pack/(Compressed)ImageLabelStamped)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.request            = msg

        if self.request.data.trueId < 0:
            self.GROUND_TRUTH   = False

        self.ego                = [self.request.data.odom.x, self.request.data.odom.y, self.request.data.odom.z]

        if self.COMPRESS_IN:
            self.store_query    = self.bridge.compressed_imgmsg_to_cv2(self.request.queryImage, "passthrough")
        else:
            self.store_query    = self.bridge.imgmsg_to_cv2(self.request.queryImage, "passthrough")

        self.new_query        = True

    def clean_cal_data(self):
        # Goals: 
        # 1. Reshape calref to match length of calqry
        # 2. Reorder calref to match 1:1 indices with calqry
        calqry_xy = np.transpose(np.stack((self.odom_calqry['position']['x'],self.odom_calqry['position']['y'])))
        calref_xy = np.transpose(np.stack((self.odom_calref['position']['x'],self.odom_calref['position']['y'])))
        match_mat = np.sum((calqry_xy[:,np.newaxis] - calref_xy)**2, 2)
        match_min = np.argmin(match_mat, 1) # should have the same number of rows as calqry (but as a vector)
        calref_xy = calref_xy[match_min, :]
        self.features_calref = self.features_calref[match_min, :]
        diff = np.sqrt(np.sum(np.square(calref_xy - calqry_xy),1))
        self.actual_match_cal = np.arange(len(self.features_calqry))

    def calibrate(self):
        self.features_calqry                = np.array(self.cal_qry_ip.SET_DICT['img_feats'][self.CAL_QRY_FOLDER])
        self.features_calref                = np.array(self.cal_ref_ip.SET_DICT['img_feats'][self.CAL_REF_FOLDER])
        self.odom_calqry                    = self.cal_qry_ip.SET_DICT['odom']
        self.odom_calref                    = self.cal_ref_ip.SET_DICT['odom']
        self.clean_cal_data()
        self.Scal, self.rmean, self.rstd    = create_normalised_similarity_matrix(self.features_calref, self.features_calqry)

    def train(self):
        # We define the acceptable tolerance for a 'correct' match as +/- one image frame:
        self.tolerance      = 10

        # Extract factors that describe the "sharpness" of distance vectors
        self.factor1_cal    = find_va_factor(self.Scal)
        self.factor2_cal    = find_grad_factor(self.Scal)
        self.factors = ['VA ratio', 'Average Gradient'] # for axis labels when plotting

        # Form input vector
        self.Xcal           = np.c_[self.factor1_cal, self.factor2_cal]
        self.scaler         = StandardScaler()
        self.Xcal_scaled    = self.scaler.fit_transform(self.Xcal)
        
        # Form desired output vector
        self.y_cal          = find_y(self.Scal, self.actual_match_cal, self.tolerance)

        # Define and train the Support Vector Machine
        self.model          = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        self.model.fit(self.Xcal_scaled, self.y_cal)

        # Make predictions on calibration set to assess performance
        self.y_pred_cal     = self.model.predict(self.Xcal_scaled)
        self.y_zvalues_cal  = self.model.decision_function(self.Xcal_scaled)

        rospy.loginfo('Performance of prediction on Calibration set: ')
        find_prediction_performance_metrics(self.y_pred_cal, self.y_cal, verbose=True)

        # Generate decision function matrix:
        f1 = np.linspace(0, factor1.max(),SIZE)
        f2 = np.linspace(0, factor2.max(),SIZE)


def exit(self):
    rospy.loginfo("Quit received.")
    sys.exit()

def main_loop(nmrc):

    if not (nmrc.new_query and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    rospy.loginfo("Query received.")
    nmrc.new_query = False
    nmrc.main_ready = False

    sequence = (nmrc.request.data.dvc - nmrc.rmean) / nmrc.rstd # normalise using parameters from the reference set
    factor_x = find_va_factor(np.c_[sequence])
    factor_y = find_grad_factor(np.c_[sequence])
    Xrt = np.c_[factor_x,factor_y]      # put the two factors into a 2-column vector
    Xrt_scaled = nmrc.scaler.transform(Xrt)  # perform scaling using same parameters as calibration set
    y_pred_rt = nmrc.model.predict(Xrt_scaled)[0] # Make the prediction: predict whether this match is good or bad

    if nmrc.request.data.state == 0:
        rospy.loginfo('integrity prediction: %s', y_pred_rt)
    else:
        gt_state_bool = bool(nmrc.request.data.state - 1)
        if y_pred_rt == gt_state_bool:
            rospy.loginfo('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))
        elif y_pred_rt == False and gt_state_bool == True:
            rospy.logwarn('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))
        else:
            rospy.logerr('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))

def do_args():
    parser = ap.ArgumentParser(prog="vpr_monitor", 
                                description="ROS implementation of Helen Carson's Integrity Monitor, for integration with QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Positional Arguments:
    parser.add_argument('ref-dataset-name', help='Specify name of reference dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('cal-qry-dataset-name', help='Specify name of calibration query dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('cal-ref-dataset-name', help='Specify name of calibration reference dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('database-path', help="Specify path to where compressed databases exist (for fast loading).")
    parser.add_argument('image-topic-in', help="Specify input image topic (exclude /compressed).")
    parser.add_argument('odometry-topic-in', help="Specify input odometry topic (exclude /compressed).")

    # Optional Arguments:
    parser.add_argument('--compress-in', '-Ci', type=check_bool, default=False, help='Enable image compression on input (default: %(default)s)')
    parser.add_argument('--compress-out', '-Co', type=check_bool, default=False, help='Enable image compression on output (default: %(default)s)')
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
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()

        # Hand to class ...
        nmrc = mrc(args['ref-dataset-name'], args['cal-qry-dataset-name'], args['cal-ref-dataset-name'], args['database-path'], args['image-topic-in'], args['odometry-topic-in'], \
                    compress_in=args['compress_in'], compress_out=args['compress_out'], \
                    rate_num=args['rate'], ft_type=enum_get(args['ft_type'], FeatureType), img_dims=args['img_dims'], icon_settings=args['(size, distance)'], \
                    tolerance_threshold=args['tol_thresh'], tolerance_mode=enum_get(args['tol_mode'], Tolerance_Mode), \
                    match_metric='euclidean', namespace=args['namespace'], \
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
