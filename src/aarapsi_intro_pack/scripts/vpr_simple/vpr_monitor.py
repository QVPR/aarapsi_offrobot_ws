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
import sys
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse as ap

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from aarapsi_intro_pack.vpred import *
from aarapsi_intro_pack import Tolerance_Mode, VPRImageProcessor, FeatureType
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get
from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool

class mrc: # main ROS class
    def __init__(self, ref_dataset_name, cal_dataset_name, database_path, image_feed_input, odometry_input, \
                    compress_in=True, compress_out=False, \
                    rate_num=20.0, ft_type=FeatureType.RAW, img_dims=(64,64), icon_settings=(50,20), \
                    tolerance_threshold=5.0, tolerance_mode=Tolerance_Mode.METRE_LINE, \
                    match_metric='euclidean', namespace="/vpr_nodes", \
                    time_history_length=20, frame_id="base_link", \
                    node_name='vpr_monitor', anon=True,\
                    cal_query_folder='forward', cal_reference_folder='forward_corrected',\
                    ref_reference_folder='forward_corrected'\
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
        self.CAL_DATA_NAME      = cal_dataset_name

        self.CAL_QRY_FOLDER     = cal_query_folder
        self.CAL_REF_FOLDER     = cal_reference_folder
        self.REF_REF_FOLDER     = ref_reference_folder

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
        rospy.logdebug("Loading calibration data set...")
        self.cal_ip             = VPRImageProcessor()
        if not self.cal_ip.npzLoader(self.DATABASE_PATH, self.CAL_DATA_NAME):
            self.exit()

        self.calibrate()
        self.train()
    
    def calibrate(self):
        #print(self.cal_ip.image_features.keys())
        self.features_calref     = self.cal_ip.image_features[self.CAL_REF_FOLDER]
        self.features_calqry     = self.cal_ip.image_features[self.CAL_QRY_FOLDER]
        self.actual_match_cal    = np.arange(0, len(self.features_calqry))
        self.Scal, _, _          = create_normalised_similarity_matrix(self.features_calref, self.features_calqry)

        fig, axes = plt.subplots(1,1,figsize=(10,6))
        imshow_handle = axes.imshow(self.Scal)
        axes.set_title('Calibration Set Distance Matrix')
        axes.set_xlabel('query frame'); axes.set_ylabel('reference frame')
        fig.show()
        imshow_handle.set_data(self.Scal)

        fig.canvas.draw() # NEEDED
        fig.canvas.draw() # >> two required to draw with no pause

    def train(self):
        # We define the acceptable tolerance for a 'correct' match as +/- one image frame:
        self.tolerance      = 1

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
        self.model          = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
        self.model.fit(self.Xcal_scaled, self.y_cal)

        # Make predictions on calibration set to assess performance
        self.y_pred_cal     = self.model.predict(self.Xcal_scaled)
        self.y_zvalues_cal  = self.model.decision_function(self.Xcal_scaled)

        rospy.loginfo('Performance of prediction on Calibration set: ')
        find_prediction_performance_metrics(self.y_pred_cal, self.y_cal, verbose=True)


def exit(self):
    rospy.loginfo("Quit received.")
    sys.exit()

def main_loop(nmrc):
    pass

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="vpr_monitor", 
                                   description="ROS implementation of Helen Carson's Integrity Monitor, for integration with QVPR's VPR Primer",
                                   epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        
        # Positional Arguments:
        parser.add_argument('ref-dataset-name', help='Specify name of reference dataset (for fast loading; matches are made on names starting with provided string).')
        parser.add_argument('cal-dataset-name', help='Specify name of calibration dataset (for fast loading; matches are made on names starting with provided string).')
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
        args = vars(raw_args[0])

        # Hand to class ...
        nmrc = mrc(args['ref-dataset-name'], args['cal-dataset-name'], args['database-path'], args['image-topic-in'], args['odometry-topic-in'], \
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
