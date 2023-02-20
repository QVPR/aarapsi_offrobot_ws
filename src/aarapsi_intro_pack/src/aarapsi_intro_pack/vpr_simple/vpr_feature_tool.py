#!/usr/bin/env python3

import rospy
import numpy as np
import os
import cv2
import sys
from enum import Enum
from tqdm import tqdm
from aarapsi_intro_pack.core.file_system_tools import check_structure, scan_directory, find_shared_root

class FeatureType(Enum):
    NONE = 0
    RAW = 1
    PATCHNORM = 2

class State(Enum):
    DEBUG = "[DEBUG]"
    INFO = "[INFO]"
    WARN = "[WARN]"
    ERROR = "[!ERROR!]"
    FATAL = "[!!FATAL!!]"

class VPRImageProcessor: # main ROS class
    def __init__(self):

        self.clearImageVariables()
        self.clearOdomVariables()

    def print(self, text, state):
        if rospy.core.is_initialized():
            if state == State.DEBUG:
                rospy.logdebug(text)
            elif state == State.INFO:
                rospy.loginfo(text)
            elif state == State.WARN:
                rospy.logwarn(text)
            elif state == State.ERROR:
                rospy.logerr(text)
            elif state == State.FATAL:
                rospy.logfatal(text)
        else:
            print(state.value + " " + text)

    def loadFull(self, img_path, odom_path, feat_type, img_dims):
        self.print("[loadFull] Attempting to load library.", State.DEBUG)
        self.loadImageFeatures(img_path, feat_type, img_dims, skip_dirs=[odom_path])
        self.loadOdometry(odom_path)
        if not (self.IMAGES_LOADED and self.ODOM_LOADED):
            self.print("[loadFull] Terminating; load procedure failed.", State.FATAL)
            sys.exit()
        return self._img_info, self._odom

    def loadImageFeatures(self, img_path_input, feat_type, img_dims, skip_dirs=["odo"], root_separation_max=1):
        self.FEAT_TYPE = feat_type
        self.IMG_DIMS = img_dims

        try:
            if isinstance(img_path_input, list):
                _, dist, root = find_shared_root(img_path_input)
                if dist > root_separation_max:
                    raise Exception("[loadImageFeatures] Error: list provided, but roots exceed specified (or default) directory separation limit (%d)!" % (root_separation_max))
                self.EXTENDED_MODE = True
                dirs = img_path_input
                img_set_root = root
                self.print("[loadImageFeatures] Extended library specified, handling...", State.WARN)
            elif isinstance(img_path_input, str):
                self.EXTENDED_MODE, dirs = check_structure(img_path_input, ".png", at=True, skip=skip_dirs)
                img_set_root = img_path_input
            else:
                raise Exception("[loadImageFeatures] Unsupported image path input type.")
            
            if self.EXTENDED_MODE:
                self.print("[loadImageFeatures] Extended library detected.", State.WARN)
                self.image_features = {}
                self.image_paths = {}
                for img_set_list in dirs:
                    self.IMG_PATH = img_set_list
                    img_set_name = os.path.basename(img_set_list)
                    self.print("[loadImageFeatures] Loading set %s" % (img_set_name), State.INFO)
                    self.processImageDataset()
                    self.image_features[img_set_name] = list(self._img_info['fts'])
                    self.image_paths[img_set_name] = list(self._img_info['paths'])
                self.IMAGES_LOADED = True
                self.IMG_PATH = img_set_root
                return len(dirs)
            
            self.print("[loadImageFeatures] Basic library detected.", State.INFO)
            self.image_features = []
            self.image_paths = []
            self.IMG_PATH = img_set_root
            self.processImageDataset()
            self.image_features = list(self._img_info['fts'])
            self.image_paths = list(self._img_info['paths'])
            if not self.image_features: # if empty
                raise Exception("[loadImageFeatures] Output is empty. Check inputs.")
            self.IMAGES_LOADED = True
            return 1
    
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_path is a valid string, feat_type is a valid FeatureType value (not NONE!), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.clearImageVariables()
            return 0
    
    def loadOdometry(self, odom_path):
        self.ODOM_PATH = odom_path
        try:
            self.processOdomDataset()
            self.odom_x = list(self._odom['x'])
            self.odom_y = list(self._odom['y'])
            self.odom_z = list(self._odom['z'])
            self.odom_paths = list(self._odom['paths'])
            if not (self.odom_x and self.odom_y): # if empty
                raise Exception("[loadOdometry] Output is empty. Check inputs.")
            if not (self.odom_z): # may not exist
                self.print("[loadOdometry] Odometry has no z values.", State.WARN)
            self.ODOM_LOADED = True
            return self.odom_x, self.odom_y, self.odom_z
        except Exception as e:
            self.print("[loadOdometry] Unable to interpret, failed. Check variables.\nEnsure: odom_path is a valid string.\nCode: %s" % (e), State.ERROR)
            self.clearOdomVariables()
            return [], [], []
        
    def _npzLoader(self, database_path, filename):
    # Private method, embedded in public methods
        # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
        ext = os.path.splitext(filename)[-1].lower()
        if (ext == ".npz"):
            filename = filename[0:-4]
        elif (ext == ""):
            pass
        elif not (ext == ".npz"):
            raise Exception("[_npzLoader] File is not of type .npz!")
        # ensure we don't double up on a "/":
        separator = ""
        if not (database_path[-1] == "/"):
            separator = "/"
        # gemerate new full path to save to:
        fs, _, _ = scan_directory(database_path)
        file_found = False
        for fname in fs:
            if fname.startswith(filename):
                data = np.load(database_path + separator + fname, allow_pickle=True)
                file_found = True
                break
        if not file_found:
            raise Exception("[_npzLoader] No file found starting with %s in directory '%s'" % (filename, database_path))

        self.FILENAME = data['filename']

        # update class attributes:
        self.EXTENDED_MODE = data['extended']
        if self.EXTENDED_MODE:
            self._img_info = {'paths': data['image_paths'].item(), 'fts': data['ft'].item()}
            self.image_features = data['ft'].item()
            self.image_paths = data['image_paths'].item()
        else:
            self._img_info = {'paths': data['image_paths'], 'fts': data['ft']}
            self.image_features = data['ft']
            self.image_paths = data['image_paths']

        self._odom = {'paths': data['odom_paths'], 'x': data['x'], 'y': data['y'], 'z': data['z']}

        self.odom_x = data['x']
        self.odom_y = data['y']
        self.odom_z = data['z']
        self.odom_paths = data['odom_paths']

        self.FEAT_TYPE = data['fttype']
        self.IMG_DIMS = data['imgdims']

        self.ODOM_LOADED = len(self.odom_paths) > 0
        self.IMAGES_LOADED = len(self.image_paths) > 0

        self.ODOM_PATH = data['odom_root']
        self.IMG_PATH = data['image_root']

        return data.keys()

    def npzLoader(self, database_path, filename):
        try:
            data_keys = self._npzLoader(database_path, filename)
            key_string = str(np.fromiter(data_keys, (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            self.print("[npzLoader] Success. Data with internal meta-filename '%s' found with keys: \n\t%s" % (self.FILENAME, key_string), State.INFO)
            if self.EXTENDED_MODE == True:
                key_string_extended = str(np.fromiter(self.image_paths.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
                self.print("[npzLoader] Data is an extended set, with image features and paths dictionaries, keys: \n\t%s" % (key_string_extended), State.WARN)
            return True
        except Exception as e:
            self.print("[npzLoader] Load failed. Check path and file name is correct.\nCode: %s" % (e), State.ERROR)
            return False

    def npzDatabaseLoadSave(self, database_path, filename, img_path, odom_path, feat_type, img_dims, do_save=False):
        load_status = self.npzLoader(database_path, filename)
        if not load_status:
            self.print("[npzDatabaseLoadSave] Load failed. Building normally.", State.WARN)
            self._img_info, self._odom = self.loadFull(img_path, odom_path, feat_type, img_dims)
        
            if do_save:
                self.print("[npzDatabaseLoadSave] Build success.", State.INFO)
                self.save2npz(database_path, filename)
        return self._img_info, self._odom
    
    def save2npz(self, database_path, filename):
        # check we have stuff to save:
        if not (self.IMAGES_LOADED):
            raise Exception("[save2npz] No images loaded. loadImageFeatures() must be performed before any save process can be performed.")
        if not (self.ODOM_LOADED):
            self.print("[save2npz] No odometry loaded: rows will be empty.", State.WARN)
        # ensure we don't double up on a ".npz":
        ext = os.path.splitext(filename)[-1].lower()
        if (ext == ".npz"):
            filename = filename[0:-4]
        # ensure we don't double up on a "/":
        separator = ""
        if not (database_path[-1] == "/"):
            separator = "/"
        ext_mode = ""
        if self.EXTENDED_MODE:
            ext_mode = "_extended"
        # gemerate new full path to save to:
        full_file_path = database_path + separator + filename + "_" + str(self.IMG_DIMS[1]) + ext_mode + ".npz"
        # perform save to compressed numpy file:
        try:
            self.print("[save2npz] Saving data as '%s'." % (full_file_path), State.INFO)
            np.savez(full_file_path, filename=full_file_path, ft=self.image_features, x=self.odom_x, y=self.odom_y, z=self.odom_z, \
                        odom_paths=self.odom_paths, image_paths=self.image_paths, fttype=self.FEAT_TYPE, imgdims=self.IMG_DIMS, \
                        odom_root=self.ODOM_PATH, image_root=self.IMG_PATH, extended=self.EXTENDED_MODE)
        except Exception as e:
            self.print("[save2npz] Unable to perform save operation. Check path.\nCode: %s" % (e), State.ERROR)

    def clearImageVariables(self):
        self.IMG_PATH           = ""
        self.FEAT_TYPE          = FeatureType.NONE
        self.IMG_DIMS           = (0,0)
        self.image_features     = []
        self.IMAGES_LOADED      = False
        self.EXTENDED_MODE      = False

    def clearOdomVariables(self):
        self.ODOM_PATH          = ""
        self.odom_x             = []
        self.odom_y             = []
        self.odom_z             = []
        self.ODOM_LOADED        = False

    def processImageDataset(self): 
    # Extract images and their features from path
    # Store in arrays and return them.

        self.print("[processImageDataset] Attempting to process images from: %s" % (self.IMG_PATH), State.DEBUG)
        imPath_list = np.sort(os.listdir(self.IMG_PATH))
        imPath_list = [os.path.join(self.IMG_PATH, f) for f in imPath_list]

        self._img_info = {'paths': imPath_list, 'fts': []}

        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
                self._img_info['fts'].append(feat)
        else:
            raise Exception("[processImageDataset] No files at path - cannot continue.")

    def getFeat(self, im, size=1, dims=None, fttype=None):
    # Get features from im, using VPRImageProcessor's set image dimensions and feature type (from loadImageFeatures).
    # Can override the dimensions and feature type using fttype= (from FeatureType enum) and dims= (two-element positive integer tuple)
    # Returns feature arary, as a flattened array (size=1) or a flattened array reshaped to 2d matrix format (size=2).

        if dims is None:
            if not (self.IMG_DIMS[0] > 0 and self.IMG_DIMS[1] > 0):
                raise Exception("[getFeat] Image dimension not set!")
            else:
                dims = self.IMG_DIMS
        if fttype is None:
            if self.FEAT_TYPE == FeatureType.NONE:
                raise Exception("[getFeat] Feature type not set!")
            else:
                fttype = self.FEAT_TYPE
        if not ( isinstance(size, int) and (size in [1, 2]) ):
            raise Exception("[getFeat] Size must be either integer 1 or 2.")
        try:
            im = cv2.resize(im, dims)
            ft = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if self.FEAT_TYPE == FeatureType.RAW:
                pass # already done
            elif self.FEAT_TYPE == FeatureType.PATCHNORM:
                ft = self.patchNormaliseImage(ft, 8)
            if size == 1:
                ft_ready = ft.flatten() # np 1d matrix format
            if size == 2: # 2d matrix
                ft_ready = ft.flatten().reshape(1,-1) # np 2D matrix format
            return ft_ready
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))

    def patchNormaliseImage(self, img, patchLength):
    # TODO: vectorize
    # take input image, divide into regions, normalise
    # returns: patch normalised image

        img1 = img.astype(float)
        img2 = img1.copy()
        
        if patchLength == 1: # single pixel; already p-n'd
            return img2

        for i in range(img1.shape[0]//patchLength): # floor division -> number of rows
            iStart = i*patchLength
            iEnd = (i+1)*patchLength
            for j in range(img1.shape[1]//patchLength): # floor division -> number of cols
                jStart = j*patchLength
                jEnd = (j+1)*patchLength

                mean1 = np.mean(img1[iStart:iEnd, jStart:jEnd])
                std1 = np.std(img1[iStart:iEnd, jStart:jEnd])

                img2[iStart:iEnd, jStart:jEnd] = img1[iStart:iEnd, jStart:jEnd] - mean1 # offset remove mean
                if std1 == 0:
                    std1 = 0.1
                img2[iStart:iEnd, jStart:jEnd] /= std1 # crush by std

        return img2
    
    def processOdomDataset(self):
    # Extract from position .csvs at path, robot x,y,z
    # Return these as nice lists

        self.print("[processOdomDataset] Attempting to process odometry from: %s" % (self.ODOM_PATH), State.DEBUG)
        odomPath_list = np.sort(os.listdir(self.ODOM_PATH))
        odomPath_list = [os.path.join(self.ODOM_PATH, f) for f in odomPath_list]
    
        self._odom = {'paths': odomPath_list, 'x': [], 'y': [], 'z': []}
        if len(odomPath_list) > 0:
            for i, odomPath in tqdm(enumerate(odomPath_list)):
                new_odom = np.loadtxt(odomPath, delimiter=',')
                self._odom['x'].append(new_odom[0])
                self._odom['y'].append(new_odom[1])
                self._odom['z'].append(new_odom[2])
        else:
            raise Exception("[processOdomDataset] No files at path - cannot continue.")

### Example usage:

# import rospkg

# PACKAGE_NAME    = 'aarapsi_intro_pack'
# SET_NAME        = "cw_loop"
# FEAT_TYPE       = FeatureType.RAW # Feature Type
# IMG_DIMS        = (64, 64)
# REF_IMG_PATH    = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/forward"
# REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odo"
# DATABASE_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"

# test = VPRImageProcessor()
# # test.loadImageFeatures(REF_IMG_PATH, FEAT_TYPE, IMG_DIMS)
# # test.loadOdometry(REF_ODOM_PATH)
# # test.save2npz(DATABASE_PATH, SET_NAME)
# # data = test.npzLoader(DATABASE_PATH, SET_NAME)
# ref_info, ref_odom = test.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME, REF_IMG_PATH, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)