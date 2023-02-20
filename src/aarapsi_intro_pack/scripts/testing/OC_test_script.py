#!/usr/bin/env python3

import rospkg
from aarapsi_intro_pack import FeatureType, VPRImageProcessor
from aarapsi_intro_pack.core.file_system_tools import *
import numpy as np
import os

PACKAGE_NAME    = 'aarapsi_intro_pack'
SET_NAME        = "cw_zeroed_20230208"
FEAT_TYPE       = FeatureType.RAW # Feature Type
IMG_DIMS        = (64, 64)
REF_IMG_ROOT    = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME
REF_IMG_PATH    = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/forward"
REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odo"
DATABASE_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"

