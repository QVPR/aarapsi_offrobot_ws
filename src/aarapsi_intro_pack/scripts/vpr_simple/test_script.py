#!/usr/bin/env python3

import rospkg
from aarapsi_intro_pack import FeatureType, VPRImageProcessor

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

def scan_directory(path):
    dir_scan = os.scandir(path) # https://docs.python.org/3/library/os.html#os.scandir
    fil_names = []
    dir_names = []
    file_exts = []
    for entry in dir_scan:
        if entry.is_file():
            if entry.name.startswith('.'):
                continue
            fil_names.append(entry.name)
            file_exts.append(os.path.splitext(entry)[-1].lower())
        elif entry.is_dir(): 
            dir_names.append(entry.name)
        else: raise Exception("Unknown file type detected.")
    return fil_names, dir_names, np.unique(file_exts)

def check_dir_type(path, filetype=None, alltype=False):
    fs, ds, exts = scan_directory(path)
    if filetype == '': filetype = None
    if (len(fs) > 0) and not (filetype is None): # contains a file and we want a file
        if (filetype in exts): # we want a file and it exists
            if (alltype == False): # we don't care if there are other file types present:
                return True
            elif (len(exts) == 1): # there is only one and it is only what we want
                return True
        return False # we wanted a file and it didn't exist, or there were multiple types and we didn't want that
    if (len(ds) > 0) and (filetype is None): # contains a folder and we only want folders
        return True
    return False # either: 
                 # 1. the directory has no files but we want a filetype
                 # 2. the directory has files but we want no files
                 # 3. the directory has no files and we want no files (good), but it has no folders either (and we want folders)

def check_structure(root, filetype, alltype=False):
    pass

test = VPRImageProcessor()
fs, ds, exts = scan_directory(REF_IMG_PATH)
path = REF_IMG_PATH
print(path)
print(check_dir_type(path, filetype=".png", alltype=False))

# # test.loadImageFeatures(REF_IMG_PATH, FEAT_TYPE, IMG_DIMS)
# # test.loadOdometry(REF_ODOM_PATH)
# # test.save2npz(DATABASE_PATH, SET_NAME)
# # data = test.npzLoader(DATABASE_PATH, SET_NAME)
# ref_info, ref_odom = test.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME, REF_IMG_PATH, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)