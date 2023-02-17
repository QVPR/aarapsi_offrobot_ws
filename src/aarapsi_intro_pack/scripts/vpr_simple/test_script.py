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

print(check_structure(REF_IMG_PATH, ".png", at=True, skip=[REF_ODOM_PATH]))
print(check_structure(REF_IMG_ROOT, ".png", at=True, skip=[REF_ODOM_PATH]))

# # # test.loadImageFeatures(REF_IMG_PATH, FEAT_TYPE, IMG_DIMS)
# # # test.loadOdometry(REF_ODOM_PATH)
# # # test.save2npz(DATABASE_PATH, SET_NAME)
# # # data = test.npzLoader(DATABASE_PATH, SET_NAME)
# # ref_info, ref_odom = test.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME, REF_IMG_PATH, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)