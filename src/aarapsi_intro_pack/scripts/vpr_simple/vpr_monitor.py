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
                                    ImageDetails, CompressedImageDetails, CompressedMonitorDetails# Our custom structures
from aarapsi_intro_pack.srv import GetSVMField, GetSVMFieldResponse
from aarapsi_intro_pack import VPRImageProcessor, SVMModelProcessor, Tolerance_Mode, FeatureType
from aarapsi_intro_pack.vpred import *
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get
from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_bool
from aarapsi_intro_pack.core.helper_tools import formatException

class mrc: # main ROS class
    def __init__(self, ref_dataset_name, cal_qry_dataset_name, cal_ref_dataset_name, database_path, image_feed_input, odometry_input, \
                    compress_in=True, compress_out=False, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    namespace="/vpr_nodes", node_name='vpr_monitor', anon=True, frame_id='base_link', \
                    cal_folder='forward', ref_folder='forward', \
                    print_prediction=True, log_level=2\
                ):

        rospy.init_node(node_name, anonymous=anon, log_level=log_level)
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

        self.CAL_FOLDER         = cal_folder
        self.REF_FOLDER         = ref_folder

        self.NAMESPACE          = namespace
        self.NODENAME           = node_name
        self.FEED_TOPIC         = image_feed_input
        self.ODOM_TOPIC         = odometry_input

        self.TOL_MODE           = tolerance_mode
        self.TOL_THRES          = tolerance_threshold

        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN        = compress_in
        self.COMPRESS_OUT       = compress_out

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        if self.COMPRESS_IN:
            self.in_img_tpc_mode    = "/compressed"
            self.in_image_type      = CompressedImage
            self.in_label_type      = CompressedImageLabelStamped
            self.in_img_dets        = CompressedImageDetails
            self.in_mon_dets        = CompressedMonitorDetails
        else:
            self.in_img_tpc_mode    = ""
            self.in_image_type      = Image
            self.in_label_type      = ImageLabelStamped
            self.in_img_dets        = ImageDetails
            self.in_mon_dets        = MonitorDetails

        if self.COMPRESS_OUT:
            self.out_img_tpc_mode   = "/compressed"
            self.out_image_type     = CompressedImage
            self.out_label_type     = CompressedImageLabelStamped
            self.out_img_dets       = CompressedImageDetails
            self.out_mon_dets       = CompressedMonitorDetails
        else:
            self.out_img_tpc_mode   = ""
            self.out_image_type     = Image
            self.out_label_type     = ImageLabelStamped
            self.out_img_dets       = ImageDetails
            self.out_mon_dets       = MonitorDetails


        ## Process reference data (only needs to be done once)
        # rospy.logdebug("Loading reference data set...")
        # self.ref_ip                 = VPRImageProcessor(ros=True)
        # if not self.ref_ip.npzLoader(self.DATABASE_PATH, self.REF_DATA_NAME, self.IMG_DIMS):
        #    self.exit()
        ## Currently reference data set is unused / unnecessary
        ## Kept in case we want to do some comparisons to the calibration data
        ## Commented out to reduce initialisation time.

        self.vpr_label_sub          = rospy.Subscriber(self.NAMESPACE + "/label" + self.in_img_tpc_mode, self.in_label_type, self.label_callback, queue_size=1)
        self.svm_state_pub          = rospy.Publisher(self.NAMESPACE + "/monitor/state" + self.in_img_tpc_mode, self.out_mon_dets, queue_size=1)
        self.svm_field_pub          = rospy.Publisher(self.NAMESPACE + "/monitor/field" + self.in_img_tpc_mode, self.out_img_dets, queue_size=1)
        self.svm_field_srv          = rospy.Service(self.NAMESPACE + '/GetSVMField', GetSVMField, self.handle_GetSVMField)

        # flags to denest main loop:
        self.new_label              = False # new label received
        self.main_ready             = False # ensure pubs and subs don't go off early

        self.last_time              = rospy.Time.now()
        self.time_history           = []

        self.states                 = [0,0,0]

        # Set up SVM
        self.svm_model_params       = dict(ref=self.CAL_REF_DATA_NAME, qry=self.CAL_QRY_DATA_NAME, img_dims=self.IMG_DIMS, \
                                           folder=self.CAL_FOLDER, database_path=self.DATABASE_PATH)
        self.svm_model_dir          = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/cfg/svm_models"
        self.svm                    = SVMModelProcessor(self.svm_model_dir, model=self.svm_model_params, ros=True)
        self.main_ready = True

    def handle_GetSVMField(self, req):
    # /vpr_nodes/GetSVMField service
        ans = GetSVMFieldResponse()
        success = True

        try:
            if req.generate == True:
                self.generate_svm_mat()
            self.svm_field_pub.publish(self.SVM_FIELD_MSG)
        except:
            success = False

        ans.success = success
        ans.topic = self.NAMESPACE + "/monitor/field" + self.in_img_tpc_mode
        rospy.logdebug("Service requested [Gen=%s], Success=%s" % (str(req.generate), str(success)))
        return ans

    def label_callback(self, msg):
    # /vpr_nodes/label(/compressed) (aarapsi_intro_pack/(Compressed)ImageLabelStamped)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.label            = msg
        self.new_label        = True

    def generate_svm_mat(self):
        # Generate decision function matrix for ros:
        array_dim = 1000
        (img_np_raw, (x_lim, y_lim)) = self.svm.generate_svm_mat(array_dim)
        img_np = np.flip(img_np_raw, axis=2) # to bgr format, for ROS

        # extract only plot region; ditch padded borders; resize to 1000x1000
        indices_cols = np.arange(img_np.shape[1])[np.sum(np.sum(img_np,2),0) != 255*3*img_np.shape[0]]
        indices_rows = np.arange(img_np.shape[0])[np.sum(np.sum(img_np,2),1) != 255*3*img_np.shape[1]]
        img_np_crop = img_np[min(indices_rows) : max(indices_rows)+1, \
                             min(indices_cols) : max(indices_cols)+1]
        img_np_crop_resize = cv2.resize(img_np_crop, (array_dim, array_dim), interpolation = cv2.INTER_AREA)
        
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
        self.SVM_FIELD_MSG.data.xlab                = 'VA ratio'
        self.SVM_FIELD_MSG.data.ylab                = 'Average Gradient'
        self.SVM_FIELD_MSG.data.title               = 'SVM Decision Function'
        self.SVM_FIELD_MSG.header.frame_id          = self.FRAME_ID
        self.SVM_FIELD_MSG.header.stamp             = rospy.Time.now()

    def exit(self):
        rospy.loginfo("Quit received.")
        sys.exit()

def main_loop(nmrc):

    if not (nmrc.new_label and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "[%s] Waiting for a new label." % (nmrc.NODENAME)) # print every 60 seconds
        return

    nmrc.new_label = False
    (y_pred_rt, y_zvalues_rt, [factor1_qry, factor2_qry]) = nmrc.svm.predict(nmrc.label.data.dvc)

    if nmrc.PRINT_PREDICTION:
        if nmrc.label.data.state == 0:
            rospy.loginfo('integrity prediction: %s', y_pred_rt)
        else:
            gt_state_bool = bool(nmrc.label.data.state - 1)
            if y_pred_rt == gt_state_bool:
                rospy.loginfo('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))
            elif y_pred_rt == False and gt_state_bool == True:
                rospy.logwarn('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))
            else:
                rospy.logerr('integrity prediction: %r [gt: %r]' % (y_pred_rt, gt_state_bool))

    # Populate and publish SVM State details
    ros_msg                 = nmrc.out_mon_dets()
    ros_msg.queryImage      = nmrc.label.queryImage
    ros_msg.header.stamp    = rospy.Time.now()
    ros_msg.header.frame_id	= str(nmrc.FRAME_ID)
    ros_msg.data            = nmrc.label.data
    ros_msg.mState	        = y_zvalues_rt # Continuous monitor state estimate 
    ros_msg.prob	        = 0.0 # Monitor probability estimate
    ros_msg.mStateBin       = y_pred_rt# Binary monitor state estimate
    ros_msg.factors         = [factor1_qry, factor2_qry]

    nmrc.svm_state_pub.publish(ros_msg)

def do_args():
    parser = ap.ArgumentParser(prog="vpr_monitor.py", 
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
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")
    
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
                    print_prediction=args['print_prediction'], log_level=args['log_level']\
                )

        rospy.loginfo("Initialisation complete. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        print("Exit state reached.")

    except rospy.ROSInterruptException:
        pass

