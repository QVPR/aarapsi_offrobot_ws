#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import rospkg
import argparse as ap
import os

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, \
    ImageDetails, CompressedImageDetails # Our custom msg structures
from aarapsi_intro_pack import VPRImageProcessor, FeatureType, \
    doDVecFigBokeh, doOdomFigBokeh, doCntrFigBokeh, updateDVecFigBokeh, updateOdomFigBokeh, updateCntrFigBokeh

from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_bool
from aarapsi_intro_pack.core.helper_tools import formatException

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

class mrc: # main ROS class
    def __init__(self, database_path, dataset_name, compress_in=True, rate_num=20.0, namespace='/vpr_nodes', node_name='vpr_all_in_one', anon=True):

        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        ## Parse all the inputs:
        self.rate_num               = rate_num # Hz
        self.rate_obj               = rospy.Rate(self.rate_num)

        self.DATABASE_PATH          = database_path
        self.REF_DATA_NAME          = dataset_name

        self.NAMESPACE              = namespace

        #!# Enable/Disable Features:
        self.COMPRESS_IN            = compress_in

        self.ego                    = [0.0, 0.0, 0.0] # robot position
        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        # Handle ROS details for input topics:
        if self.COMPRESS_IN:
            self.in_img_tpc_mode    = "/compressed"
            self.in_image_type      = CompressedImage
            self.in_label_type      = CompressedImageLabelStamped
            self.in_img_dets        = CompressedImageDetails
        else:
            self.in_img_tpc_mode    = ""
            self.in_image_type      = Image
            self.in_label_type      = ImageLabelStamped
            self.in_img_dets        = ImageDetails

        self.vpr_label_sub          = rospy.Subscriber(self.NAMESPACE + "/label" + self.in_img_tpc_mode, self.in_label_type, self.label_callback, queue_size=1)
        self.svm_field_sub          = rospy.Subscriber(self.NAMESPACE + "/monitor/field", self.in_img_dets, queue_size=1)

        # flags to denest main loop:
        self.new_query              = False # new query image (MAKE_LABEL==True) or new label received (MAKE_LABEL==False)
        self.main_ready             = False

        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor()
        self.image_processor.npzLoader(self.DATABASE_PATH, self.REF_DATA_NAME)
        self.ref_dict               = self.image_processor.SET_DICT

        # Prepare figures:
        #self.fig.suptitle("Odometry Visualised")
        iframe_start          = """<iframe src="http://131.181.33.60:8080/stream?topic="""
        iframe_end_rect       = """&type=ros_compressed" width=2000 height=1000 style="border: 0; transform: scale(0.5); transform-origin: 0 0;"/>"""
        iframe_end_even       = """&type=ros_compressed" width=510 height=510 style="border: 0; transform: scale(0.49); transform-origin: 0 0;"/>"""
        self.fig_iframe_feed_ = Div(text=iframe_start + """/vpr_nodes/image""" + iframe_end_rect, width=500, height=250)
        #self.fig_iframe_frwd_ = Div(text=iframe_start + """/ros_indigosdk_occam/image0""" + iframe_end_rect, width=500, height=250)
        self.fig_iframe_mtrx_ = Div(text=iframe_start + """/vpr_nodes/matrices/rolling""" + iframe_end_even, width=250, height=250)
        #self.fig_iframe_field = Div(text=iframe_start + """/vpr_nodes/monitor/field""" + iframe_end_even, width=250, height=250)
        self.rolling_mtrx_img = np.zeros((len(self.ref_dict['odom']['position']['x']), len(self.ref_dict['odom']['position']['x']))) # Make similarity matrix figure
        self.fig_dvec_handles = doDVecFigBokeh(self, self.ref_dict['odom']) # Make distance vector figure
        self.fig_odom_handles = doOdomFigBokeh(self, self.ref_dict['odom']) # Make odometry figure
        self.fig_cntr_handles = doCntrFigBokeh(self, self.ref_dict['odom']) # Make contour figure

        # Last item as it sets a flag that enables main loop execution.
        self.main_ready = True

    def label_callback(self, msg):
    # /vpr_nodes/label(/compressed) (aarapsi_intro_pack/(Compressed)ImageLabelStamped)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback
        self.request            = msg
        self.ego                = [msg.data.odom.x, msg.data.odom.y, msg.data.odom.z]

        if self.COMPRESS_IN:
            self.store_query    = self.bridge.compressed_imgmsg_to_cv2(self.request.queryImage, "passthrough")
        else:
            self.store_query    = self.bridge.imgmsg_to_cv2(self.request.queryImage, "passthrough")

        self.new_query          = True

def main_loop(nmrc):
# Main loop process

    if not (nmrc.new_query and nmrc.main_ready): # denest
        return

    dvc             = np.transpose(np.matrix(nmrc.request.data.dvc))
    matchInd        = nmrc.request.data.matchId
    trueInd         = nmrc.request.data.trueId

    # Clear flags:
    nmrc.new_query      = False
    nmrc.new_odom       = False

    # Update odometry visualisation:
    updateDVecFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateOdomFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateCntrFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])

def ros_spin(nmrc):
    
    nmrc.rate_obj.sleep()
    main_loop(nmrc)

    if rospy.is_shutdown():
        global server
        server.io_loop.stop()
        server.stop()
        rospy.loginfo("Exit state reached.")

def do_args():
    parser = ap.ArgumentParser(prog="vpr_all_in_one", 
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    # Positional Arguments:
    parser.add_argument('dataset-name', help='Specify name of dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('database-path', help="Specify path to where compressed databases exist (for fast loading).")

    # Optional Arguments:
    parser.add_argument('--compress-in', '-Ci', type=check_bool, default=False, help='Enable image compression on input (default: %(default)s)')
    parser.add_argument('--rate', '-r', type=check_positive_float, default=10.0, help='Set node rate (default: %(default)s).')
    parser.add_argument('--node-name', '-N', default="vpr_all_in_one", help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', default="/vpr_nodes", help="Specify namespace for topics (default: %(default)s).")

    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])
    
def main(doc):
    try:
        # Parse args...
        args = do_args()

        # Hand to class ...
        nmrc = mrc(args['database-path'], args['dataset-name'], compress_in=args['compress_in'], rate_num=args['rate'], namespace=args['namespace'], node_name=args['node_name'], anon=args['anon'])

        doc.add_root(row(   column(nmrc.fig_iframe_feed_, row(nmrc.fig_iframe_mtrx_, nmrc.fig_iframe_field)), \
                            column(nmrc.fig_dvec_handles['fig'], nmrc.fig_odom_handles['fig']), \
                            #column(nmrc.fig_cntr_handles['fig']) \
                        ))

        rospy.loginfo("Initialisation complete. Listening for queries...")

        root = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/scripts/vpr_simple/"
        doc.theme = Theme(filename=root + "theme.yaml")

        doc.add_periodic_callback(partial(ros_spin, nmrc=nmrc), int(1000 * (1/nmrc.rate_num)))
    except Exception:
        rospy.logerr(formatException())

if __name__ == '__main__':
    os.environ['BOKEH_ALLOW_WS_ORIGIN'] = '0.0.0.0:5006,131.181.33.60:5006'

    port=5006
    server = Server({'/': main}, num_procs=1, address='0.0.0.0', port=port)
    server.start()

    #print('Opening Bokeh application on http://localhost' + str(port))
    #server.show("/")
    server.io_loop.start()

    

    