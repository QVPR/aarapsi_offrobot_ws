#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import rospkg
import argparse as ap
import os
import sys
import copy

from aarapsi_intro_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, \
    ImageDetails, CompressedImageDetails, MonitorDetails, CompressedMonitorDetails # Our custom msg structures
from aarapsi_intro_pack.srv import GetSVMField, GetSVMFieldRequest, GetSVMFieldResponse
from aarapsi_intro_pack import VPRImageProcessor, FeatureType, \
    doDVecFigBokeh, doOdomFigBokeh, doCntrFigBokeh, updateDVecFigBokeh, updateOdomFigBokeh, updateCntrFigBokeh

from aarapsi_intro_pack.core.argparse_tools import check_positive_float, check_bool, check_positive_two_int_tuple, check_positive_int, check_valid_ip
from aarapsi_intro_pack.core.helper_tools import formatException, getArrayDetails
from aarapsi_intro_pack.core.enum_tools import enum_value_options, enum_get

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

import logging
logging.getLogger('bokeh').setLevel(logging.CRITICAL) # hide bokeh superfluous messages

class mrc: # main ROS class
    def __init__(self, database_path, dataset_name, ft_type=FeatureType.RAW, compress_in=True, rate_num=20.0, \
                 img_dims=(64,64), namespace='/vpr_nodes', node_name='vpr_all_in_one', \
                 anon=True, log_level=2):
        

        self.NODENAME               = node_name
        self.NAMESPACE              = namespace

        rospy.init_node(self.NODENAME, anonymous=anon, log_level=log_level)
        rospy.loginfo('Starting %s node.' % (node_name))

        # flags to denest main loop:
        self.new_state              = False # new SVM state message received
        self.new_field              = False # new SVM field message received
        self.field_exists           = False # first SVM field message hasn't been received
        self.main_ready             = False

        ## Parse all the inputs:
        self.rate_num               = rate_num # Hz
        self.rate_obj               = rospy.Rate(self.rate_num)

        self.DATABASE_PATH          = database_path
        self.REF_DATA_NAME          = dataset_name

        self.FEAT_TYPE              = ft_type
        self.IMG_DIMS               = img_dims

        #!# Enable/Disable Features:
        self.COMPRESS_IN            = compress_in

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        self._compress_on           = {'topic': "/compressed", 'image': CompressedImage, 'label': CompressedImageLabelStamped, 
                                       'img_dets': CompressedImageDetails, 'mon_dets': CompressedMonitorDetails}
        self._compress_off          = {'topic': "", 'image': Image, 'label': ImageLabelStamped, 
                                       'img_dets': ImageDetails, 'mon_dets': MonitorDetails}
        
        # Handle ROS details for input topics:
        if self.COMPRESS_IN:
            self.INPUTS             = self._compress_on
        else:
            self.INPUTS             = self._compress_off

        self.svm_state_sub          = rospy.Subscriber(self.NAMESPACE + "/monitor/state" + self.INPUTS['topic'], self.INPUTS['mon_dets'], self.state_callback, queue_size=1)
        self.svm_field_sub          = rospy.Subscriber(self.NAMESPACE + "/monitor/field" + self.INPUTS['topic'], self.INPUTS['img_dets'], self.field_callback, queue_size=1)
        self.srv_GetSVMField        = rospy.ServiceProxy(self.NAMESPACE + '/GetSVMField', GetSVMField)
        self.srv_GetSVMField_once   = False
        
        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor(ros=True, init_hybridnet=False, init_netvlad=False, cuda=False, dims=self.IMG_DIMS)
        if not self.image_processor.npzLoader(self.DATABASE_PATH, self.REF_DATA_NAME):
            exit()
        self.ref_dict               = copy.deepcopy(self.image_processor.SET_DICT) # store reference data
        self.image_processor.destroy() # destroy to save client memory

        # Prepare figures:
        iframe_start          = """<iframe src="http://131.181.33.60:8080/stream?topic="""
        iframe_end_rect       = """&type=ros_compressed" width=2000 height=1000 style="border: 0; transform: scale(0.5); transform-origin: 0 0;"/>"""
        iframe_end_even       = """&type=ros_compressed" width=510 height=510 style="border: 0; transform: scale(0.49); transform-origin: 0 0;"/>"""
        self.fig_iframe_feed_ = Div(text=iframe_start + self.NAMESPACE + "/image" + iframe_end_rect, width=500, height=250)
        self.fig_iframe_mtrx_ = Div(text=iframe_start + self.NAMESPACE + "/matrices/rolling" + iframe_end_even, width=250, height=250)
        self.rolling_mtrx_img = np.zeros((len(self.ref_dict['odom']['position']['x']), len(self.ref_dict['odom']['position']['x']))) # Make similarity matrix figure
        
        rospy.loginfo('[Bokeh Server] Waiting for services ...')
        rospy.wait_for_service(self.NAMESPACE + '/GetSVMField')
        rospy.loginfo('[Bokeh Server] Services ready.')
        
        self.fig_dvec_handles = doDVecFigBokeh(self, self.ref_dict['odom']) # Make distance vector figure
        self.fig_odom_handles = doOdomFigBokeh(self, self.ref_dict['odom']) # Make odometry figure
        self.fig_cntr_handles = doCntrFigBokeh(self, self.ref_dict['odom']) # Make contour figure

        # Last item as it sets a flag that enables main loop execution.
        self.main_ready = True
    
    def _timer_srv_GetSVMField(self, generate=False):
        try:
            if not self.main_ready:
                return
            requ = GetSVMFieldRequest()
            requ.generate = generate
            resp = self.srv_GetSVMField(requ)
            if resp.success == False:
                rospy.logerr('[timer_srv_GetSVMField] Service executed, success=False!')
            self.srv_GetSVMField_once = True
        except:
            rospy.logerr(formatException())

    def timer_srv_GetSVMField(self):
        self._timer_srv_GetSVMField(generate=True)

    def field_callback(self, msg):
    # /vpr_nodes/monitor/field(/compressed) (aarapsi_intro_pack/(Compressed)ImageDetails)
    # Store new SVM field
        self.svm_field_msg  = msg
        self.new_field      = True
        self.field_exists   = True

    def state_callback(self, msg):
    # /vpr_nodes/monitor/state(/compressed) (aarapsi_intro_pack/(Compressed)MonitorDetails)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback
        self.state              = msg
        self.new_state          = True

def main_loop(nmrc):
# Main loop process
    if not nmrc.srv_GetSVMField_once:
        nmrc._timer_srv_GetSVMField(generate=True)

    if not (nmrc.new_state and nmrc.main_ready): # denest
        return

    # Clear flags:
    nmrc.new_state  = False

    dvc             = np.transpose(np.matrix(nmrc.state.data.dvc))
    matchInd        = nmrc.state.data.matchId
    trueInd         = nmrc.state.data.trueId

    # Update odometry visualisation:
    updateDVecFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateOdomFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    if nmrc.field_exists:
        updateCntrFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])

def exit():
    global server
    server.io_loop.stop()
    server.stop()
    msg = 'Exit state reached.'
    if not rospy.is_shutdown():
        rospy.loginfo(msg)
    else:
        print(msg)
    sys.exit()

def ros_spin(nmrc):
    try:
        nmrc.rate_obj.sleep()
        main_loop(nmrc)

        if rospy.is_shutdown():
            exit()
    except:
        rospy.logerr(formatException())
        exit()

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
    parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")
    ft_options, ft_options_text = enum_value_options(FeatureType, skip=FeatureType.NONE)
    parser.add_argument('--ft-type', '-F', type=int, choices=ft_options, default=ft_options[0], \
                        help='Choose feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])
    
def main(doc):
    try:
        # Parse args...
        args = do_args()

        # Hand to class ...
        nmrc = mrc(args['database-path'], args['dataset-name'], enum_get(args['ft_type'], FeatureType), compress_in=args['compress_in'], \
                   rate_num=args['rate'], namespace=args['namespace'], img_dims=args['img_dims'], \
                    node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'])

        doc.add_root(row(   column(nmrc.fig_iframe_feed_, row(nmrc.fig_iframe_mtrx_)), \
                            column(nmrc.fig_dvec_handles['fig'], nmrc.fig_odom_handles['fig']), \
                            column(nmrc.fig_cntr_handles['fig']) \
                        ))

        rospy.loginfo("[Bokeh Server] Initialisation complete. Listening for queries...")

        root = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/scripts/vpr_simple/"
        doc.theme = Theme(filename=root + "theme.yaml")

        doc.add_periodic_callback(partial(ros_spin, nmrc=nmrc), int(1000 * (1/nmrc.rate_num)))
        #doc.add_periodic_callback(nmrc.timer_srv_GetSVMField, 1000)
    except Exception:
        rospy.logerr(formatException())
        exit()

if __name__ == '__main__':
    os.environ['BOKEH_ALLOW_WS_ORIGIN'] = '0.0.0.0,131.181.33.60'

    parser_server = ap.ArgumentParser()
    parser_server.add_argument('--port', '-P', type=check_positive_int, default=5006, help='Set bokeh server port  (default: %(default)s).')
    parser_server.add_argument('--address', '-A', type=check_valid_ip, default='0.0.0.0', help='Set bokeh server address (default: %(default)s).')
    server_vars = vars(parser_server.parse_known_args()[0])

    port=5006
    server = Server({'/': main}, num_procs=1, address=server_vars['address'], port=server_vars['port'])
    server.start()

    #print('Opening Bokeh application on http://localhost' + str(port))
    #server.show("/")
    server.io_loop.start()

## Useful
# rospy.loginfo([server._tornado._applications['/']._session_contexts[key].destroyed for key in server._tornado._applications['/']._session_contexts.keys()])

    