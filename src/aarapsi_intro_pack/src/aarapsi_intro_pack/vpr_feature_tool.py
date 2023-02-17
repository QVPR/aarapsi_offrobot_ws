#!/usr/bin/env python3
import rospy
import numpy as np
import os
import cv2
import sys
from enum import Enum
from tqdm import tqdm

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
        if not rospy.is_shutdown():
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
        self.loadImageFeatures(img_path, feat_type, img_dims)
        self.loadOdometry(odom_path)
        if not (self.IMAGES_LOADED and self.ODOM_LOADED):
            self.print("[loadFull] Terminating; load procedure failed.", State.FATAL)
            sys.exit()
        return self._img_info, self._odom

    def loadImageFeatures(self, img_path, feat_type, img_dims):
        self.IMG_PATH = img_path
        self.FEAT_TYPE = feat_type
        self.IMG_DIMS = img_dims
        try:
            self.processImageDataset()
            self.image_features = list(self._img_info['fts'])
            self.image_paths = list(self._img_info['paths'])
            if not self.image_features: # if empty
                raise Exception("[loadImageFeatures] Output is empty. Check inputs.")
            self.IMAGES_LOADED = True
            return self.image_features
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_path is a valid string, feat_type is a valid FeatureType value (not NONE!), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.clearImageVariables()
            return []
    
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
        full_file_path = database_path + separator + filename + "_" + str(self.IMG_DIMS[1]) + ".npz"

        data = np.load(full_file_path, allow_pickle=True)

        # update class attributes:
        self._img_info = {'paths': data['image_paths'], 'fts': data['ft']}
        self._odom = {'paths': data['odom_paths'], 'x': data['x'], 'y': data['y'], 'z': data['z']}

        self.odom_x = list(data['x'])
        self.odom_y = list(data['y'])
        self.odom_z = list(data['z'])
        self.odom_paths = list(data['odom_paths'])

        self.FEAT_TYPE = data['fttype']
        self.IMG_DIMS = data['imgdims']
        self.image_features = list(data['ft'])
        self.image_paths = list(data['image_paths'])

        self.ODOM_LOADED = (self.odom_paths == True)
        self.IMAGES_LOADED = (self.image_paths == True)

        self.ODOM_PATH = data['odom_root']
        self.IMG_PATH = data['image_root']

        return data

    def npzLoader(self, database_path, filename):
        try:
            data = self._npzLoader(database_path, filename)
            key_string = str(np.fromiter(data.keys(), (str, 15))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            self.print("[npzLoader] Success. Data found with keys: %s" % (key_string), State.INFO)
            return data
        except Exception as e:
            self.print("[npzLoader] Load failed. Check path and file name is correct.\nCode: %s" % (e), State.ERROR)

    def npzDatabaseLoadSave(self, database_path, filename, img_path, odom_path, feat_type, img_dims, do_save=False):
        try:
            self.IMG_DIMS = img_dims
            data = self._npzLoader(database_path, filename)
            key_string = str(np.fromiter(data.keys(), (str, 15))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            self.print("[npzDatabaseLoadSave] Success. Data with filename '%s' found with keys: %s" % (data['filename'], key_string), State.INFO)
        except Exception as e:
            self.print("[npzDatabaseLoadSave] Load failed. Building normally.\nCode: %s" % (e), State.WARN)
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
        # gemerate new full path to save to:
        full_file_path = database_path + separator + filename + "_" + str(self.IMG_DIMS[1]) + ".npz"
        # perform save to compressed numpy file:
        try:
            self.print("[save2npz] Saving data as '%s'." % (full_file_path), State.INFO)
            np.savez(full_file_path, filename=full_file_path, ft=self.image_features, x=self.odom_x, y=self.odom_y, z=self.odom_z, odom_paths=self.odom_paths, image_paths=self.image_paths, fttype=self.FEAT_TYPE, imgdims=self.IMG_DIMS, odom_root=self.ODOM_PATH, image_root=self.IMG_PATH)
        except Exception as e:
            self.print("[save2npz] Unable to perform save operation. Check path.\nCode: %s" % (e), State.ERROR)

    def clearImageVariables(self):
        self.IMG_PATH           = ""
        self.FEAT_TYPE          = FeatureType.NONE
        self.IMG_DIMS           = (0,0)
        self.image_features     = []
        self.IMAGES_LOADED      = False

    def clearOdomVariables(self):
        self.ODOM_PATH          = ""
        self.odom_x             = []
        self.odom_y             = []
        self.odom_z             = []
        self.ODOM_LOADED        = False

    def loadImageFeaturesExtended(self, img_set_root, feat_type, img_dims):
        self.IMG_PATH = img_set_root
        self.FEAT_TYPE = feat_type
        self.IMG_DIMS = img_dims
        try:
            self.processImageDataset()
            self.image_features = list(self._img_info['fts'])
            self.image_paths = list(self._img_info['paths'])
            if not self.image_features: # if empty
                raise Exception("[loadImageFeatures] Output is empty. Check inputs.")
            self.IMAGES_LOADED = True
            return self.image_features
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_path is a valid string, feat_type is a valid FeatureType value (not NONE!), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.clearImageVariables()
            return []

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