#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import numpy as np
import rospkg
import argparse as ap
import os
import sys
import cv2

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, MonitorDetails, \
                                    ImageDetails, CompressedImageDetails# Our custom structures
from aarapsi_intro_pack import VPRImageProcessor, Tolerance_Mode, FeatureType, grey2dToColourMap
from aarapsi_intro_pack.vpred import *
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get
from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool

from sklearn import svm
from sklearn.preprocessing import StandardScaler

class mrc: # main ROS class
    def __init__(self, ref_dataset_name, cal_qry_dataset_name, cal_ref_dataset_name, database_path, image_feed_input, odometry_input, \
                    compress_in=True, compress_out=False, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    namespace="/vpr_nodes", node_name='vpr_monitor', anon=True, frame_id='base_link', \
                    cal_qry_folder='forward', cal_ref_folder='forward', ref_ref_folder='forward', \
                    print_prediction=True,\
                ):

        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.rate_num           = rate_num # Hz, maximum of between 21-26 Hz (varies) with no plotting/image/ground truth/compression.
        self.rate_obj           = rospy.Rate(self.rate_num)

        self.PRINT_PREDICTION   = print_prediction

        #!# Tune Here:
        self.FEAT_TYPE          = ft_type
        self.IMG_DIMS           = img_dims
        self.FRAME_ID           = frame_id

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

        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN        = compress_in
        self.COMPRESS_OUT       = compress_out

        self.ego                = [0.0, 0.0, 0.0] # robot position

        self.bridge             = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        if self.COMPRESS_IN:
            self.in_img_tpc_mode   = "/compressed"
            self.in_image_type     = CompressedImage
            self.in_label_type     = CompressedImageLabelStamped
            self.in_img_dets       = CompressedImageDetails
        else:
            self.in_img_tpc_mode   = ""
            self.in_image_type     = Image
            self.in_label_type     = ImageLabelStamped
            self.in_img_dets       = ImageDetails

        if self.COMPRESS_OUT:
            self.out_img_tpc_mode   = "/compressed"
            self.out_image_type     = CompressedImage
            self.out_label_type     = CompressedImageLabelStamped
            self.out_img_dets       = CompressedImageDetails
        else:
            self.out_img_tpc_mode   = ""
            self.out_image_type     = Image
            self.out_label_type     = ImageLabelStamped
            self.out_img_dets       = ImageDetails

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

        self.vpr_label_sub      = rospy.Subscriber(self.NAMESPACE + "/label" + self.in_img_tpc_mode, self.in_label_type, self.label_callback, queue_size=1)
        self.svm_state_pub      = rospy.Publisher(self.NAMESPACE + "/monitor/state", MonitorDetails, queue_size=1)
        self.svm_field_pub      = rospy.Publisher(self.NAMESPACE + "/monitor/field", self.out_img_dets, queue_size=1)

        #self.svm_field_timer    = rospy.Timer(rospy.Duration(secs=1), self.svm_field_timer_cb)

        # flags to denest main loop:
        self.new_query              = False # new label received
        self.main_ready             = False # ensure pubs and subs don't go off early
        self.svm_field_pub_flag     = False # rate limiter via timer

        self.last_time              = rospy.Time.now()
        self.time_history           = []

        self.states                 = [0,0,0]

        self.calibrate()
        self.train()
        self.generate_svm_mat()

        self.main_ready = True

    # def svm_field_timer_cb(self, event):
    #     self.svm_field_pub_flag = True

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
        self.factors        = ['VA ratio', 'Average Gradient'] # for axis labels when plotting

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

    def generate_svm_mat(self):
        # Generate decision function matrix:
        array_dim   = 500 # affects performance with the trade-off on how "nice"/"smooth" it looks
        f1          = np.linspace(0, self.factor1_cal.max(), array_dim)
        f2          = np.linspace(0, self.factor2_cal.max(), array_dim)
        F1, F2      = np.meshgrid(f1, f2)
        Fscaled     = self.scaler.transform(np.vstack([F1.ravel(), F2.ravel()]).T)
        y_zvalues_t = self.model.decision_function(Fscaled).reshape([array_dim, array_dim])

        fig, ax = plt.subplots()
        ax.imshow(y_zvalues_t, origin='lower',extent=[0, f1[-1], 0, f2[-1]], aspect='auto')
        z_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0], colors='red')
        p_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0.75])
        ax.clabel(p_contour, inline=True, fontsize=8)
        x_lim = [0, self.factor1_cal.max()]
        y_lim = [0, self.factor2_cal.max()]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_box_aspect(1)
        ax.axis('off')
        fig.canvas.draw()

        # extract matplotlib canvas as an rgb image:
        img_np_raw_flat = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img_np_raw = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all')
        img_np = np.flip(img_np_raw, axis=2) # to bgr format, for ROS

        # extract only plot region; ditch padded borders
        indices_cols = np.arange(img_np.shape[1])[np.sum(np.sum(img_np,2),0) != 255*3*img_np.shape[0]]
        indices_rows = np.arange(img_np.shape[0])[np.sum(np.sum(img_np,2),1) != 255*3*img_np.shape[1]]
        img_np_crop = img_np[min(indices_rows) : max(indices_rows)+1, \
                             min(indices_cols) : max(indices_cols)+1]
        
        # keeping aspect ratio, resize up so that the longest side is now 500 pixels:
        img_new_shape = tuple((np.array(img_np_crop.shape)[0:2] * 500 / max(img_np_crop.shape)).astype(int))
        img_np_crop_resize = cv2.resize(img_np_crop, img_new_shape, interpolation = cv2.INTER_AREA)
        
        if self.COMPRESS_OUT:
            ros_img_to_pub = self.bridge.cv2_to_compressed_imgmsg(img_np_crop_resize, "jpg") # jpg (png slower)
        else:
            ros_img_to_pub = self.bridge.cv2_to_imgmsg(img_np_crop_resize, "bgr8")

        self.SVM_FIELD_MSG                          = self.out_img_dets()
        self.SVM_FIELD_MSG.image                    = ros_img_to_pub
        self.SVM_FIELD_MSG.image.header.frame_id    = self.FRAME_ID
        self.SVM_FIELD_MSG.image.header.stamp       = rospy.Time.now()
        self.SVM_FIELD_MSG.data.xlim                = x_lim
        self.SVM_FIELD_MSG.data.ylim                = y_lim
        self.SVM_FIELD_MSG.data.xlab                = self.factors[0]
        self.SVM_FIELD_MSG.data.ylab                = self.factors[1]
        self.SVM_FIELD_MSG.data.title               = 'SVM Decision Function'
        self.SVM_FIELD_MSG.header.frame_id          = self.FRAME_ID
        self.SVM_FIELD_MSG.header.stamp             = rospy.Time.now()

        self.svm_field_pub.publish(self.SVM_FIELD_MSG)
        self.svm_field_pub_flag = False

def exit(self):
    rospy.loginfo("Quit received.")
    sys.exit()

def main_loop(nmrc):

    if not (nmrc.new_query and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    nmrc.new_query = False

    sequence = (nmrc.request.data.dvc - nmrc.rmean) / nmrc.rstd # normalise using parameters from the reference set
    factor1_qry = find_va_factor(np.c_[sequence])
    factor2_qry = find_grad_factor(np.c_[sequence])
    # rt for realtime; still don't know what 'X' and 'y' mean! TODO
    Xrt = np.c_[factor1_qry, factor2_qry]      # put the two factors into a 2-column vector
    Xrt_scaled = nmrc.scaler.transform(Xrt)  # perform scaling using same parameters as calibration set
    y_zvalues_rt = nmrc.model.decision_function(Xrt_scaled) # not sure what this is just yet TODO
    y_pred_rt = nmrc.model.predict(Xrt_scaled)[0] # Make the prediction: predict whether this match is good or bad

    if nmrc.PRINT_PREDICTION:
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

    # Populate and publish SVM State details
    ros_msg                 = MonitorDetails()
    ros_msg.header.stamp    = rospy.Time.now()
    ros_msg.header.frame_id	= str(nmrc.FRAME_ID)
    ros_msg.matchId	        = nmrc.request.data.matchId# Index of VPR match
    ros_msg.trueId	        = nmrc.request.data.trueId# Index ground truth thinks is best (-1 if no groundtruth)
    ros_msg.state	        = nmrc.request.data.state# int matching 0: unknown, 1: bad, 2: good
    ros_msg.mState	        = 0.0# Continuous monitor state estimate 
    ros_msg.prob	        = 0.0# Monitor probability estimate
    ros_msg.mStateBin       = y_pred_rt# Binary monitor state estimate
    ros_msg.fieldUpdate     = nmrc.svm_field_pub_flag # Whether or not an SVM field update is needed

    nmrc.svm_state_pub.publish(ros_msg)
    if nmrc.svm_field_pub_flag:
        nmrc.svm_field_pub.publish(nmrc.SVM_FIELD_MSG)
        nmrc.svm_field_pub_flag = False

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
    parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
    ft_options, ft_options_text = enum_value_options(FeatureType, skip=FeatureType.NONE)
    parser.add_argument('--ft-type', '-F', type=int, choices=ft_options, default=ft_options[0], \
                        help='Choose feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
    tolmode_options, tolmode_options_text = enum_value_options(Tolerance_Mode)
    parser.add_argument('--tol-mode', '-t', type=int, choices=tolmode_options, default=tolmode_options[0], \
                        help='Choose tolerance mode for ground truth, types: %s (default: %s).' % (tolmode_options_text, '%(default)s'))
    parser.add_argument('--tol-thresh', '-T', type=check_positive_float, default=1.0, help='Set tolerance threshold for ground truth (default: %(default)s).')
    parser.add_argument('--node-name', '-N', default="vpr_all_in_one", help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', default="/vpr_nodes", help="Specify namespace for topics (default: %(default)s).")
    parser.add_argument('--frame-id', '-f', default="base_link", help="Specify frame_id for messages (default: %(default)s).")
    parser.add_argument('--print-prediction', '-p', type=check_bool, default=True, help="Specify whether the monitor's prediction should be printed (default: %(default)s).")

    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()

        # Hand to class ...
        nmrc = mrc(args['ref-dataset-name'], args['cal-qry-dataset-name'], args['cal-ref-dataset-name'], args['database-path'], args['image-topic-in'], args['odometry-topic-in'], \
                    compress_in=args['compress_in'], compress_out=args['compress_out'], \
                    rate_num=args['rate'], ft_type=enum_get(args['ft_type'], FeatureType), img_dims=args['img_dims'], \
                    tolerance_threshold=args['tol_thresh'], tolerance_mode=enum_get(args['tol_mode'], Tolerance_Mode), \
                    namespace=args['namespace'], node_name=args['node_name'], anon=args['anon'],  frame_id=args['frame_id'], \
                    print_prediction=args['print_prediction'], \
                )

        rospy.loginfo("Initialisation complete. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
