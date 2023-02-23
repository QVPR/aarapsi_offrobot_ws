#!/usr/bin/env python3

import rospy # comment this out if no ROS
import numpy as np
import os
import cv2
import sys
import csv
from enum import Enum
from tqdm import tqdm

# For image processing type
class FeatureType(Enum):
    NONE = 0
    RAW = 1
    PATCHNORM = 2

# For logging
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

    # def print(self, text, state):
    # # Print function helper
    # # For use with no 'import rospy'
    #     print(state.value + " " + text)

    def print(self, text, state):
    # Print function helper
    # For use with integration with ROS
        if rospy.core.is_initialized(): # if used inside of a running ROS node
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

    def buildFullDictionary(self):
        self.SET_DICT = {'img_feats': self.IMG_FEATS, \
                         'odom': self.ODOM, \
                         'times': self.TIMES,  \
                         'odom_path': self.ODOM_PATH, \
                         'image_paths': self.IMG_PATHS,  \
                         'feat_type': self.FEAT_TYPE, \
                         'img_dims': self.IMG_DIMS
                        }
    
    def loadFull(self, img_paths, odom_path, feat_type, img_dims):
    # Load in odometry and image library from raw files (.png and .csv)
    # Feed img_paths to directory containing directories of images
    # odom_path should always be full file path of an odometry.csv file
    # feat_type of enum type FeatureType
    # img_dims a two-integer positive tuple containing image width and height (width, height) in pixels
    # Returns full dictionary; empty on fail.

        self.print("[loadFull] Attempting to load library.", State.DEBUG)
        if not self.loadImageFeatures(img_paths, feat_type, img_dims, skip_dirs=[odom_path]): raise Exception("Fatal")
        if not len(self.loadOdometry(odom_path)): raise Exception("Fatal")
        self.buildFullDictionary()
        if not (self.IMAGES_LOADED and self.ODOM_LOADED):
            self.print("[loadFull] Terminating; load procedure failed.", State.FATAL)
            sys.exit()
        return self.SET_DICT

    def loadImageFeatures(self, img_paths, feat_type, img_dims, skip_dirs=["odo"]):
    # Load in images

        if not isinstance(img_paths, list):
            if isinstance(img_paths, str):
                self.print("[loadImageFeatures] img_paths is of type str and not type list. Wrapping in a list to proceed.")
                img_paths = [img_paths]
            else:
                raise Exception("img_paths provided is not of type list.")

        if not isinstance(feat_type, FeatureType):
            raise Exception("feat_type provided is not of type FeatureType.")
        
        self.FEAT_TYPE = feat_type
        self.IMG_DIMS = img_dims
        self.IMG_PATHS = img_paths

        try:
            self.IMG_FEATS = {}
            for img_set in self.IMG_PATHS:
                img_set_name = os.path.basename(img_set)
                self._IMG_DIR = img_set
                self.print("[loadImageFeatures] Loading set %s" % (img_set_name), State.INFO)
                self.processImageDataset()
                self.IMG_FEATS[img_set_name] = self._IMG_FEATS
            self.IMAGES_LOADED = True
            return len(self.IMG_PATHS)
    
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_paths is a valid string array, feat_type is a valid FeatureType value (not NONE), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.clearImageVariables()
            return 0

    def processImageDataset(self): 
    # Extract images and their features from path
    # Store in array and return them.

        self.print("[processImageDataset] Attempting to process images from: %s" % (self._IMG_DIR), State.DEBUG)
        imPath_list = np.sort(os.listdir(self._IMG_DIR))
        imPath_list = [os.path.join(self._IMG_DIR, f) for f in imPath_list]

        self._IMG_FEATS = []

        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
                self._IMG_FEATS.append(feat)
        else:
            raise Exception("[processImageDataset] No files at path - cannot continue.")
     
    def loadOdometry(self, odom_path):
    # Load in odometry from path
        self.ODOM_PATH = odom_path
        try:
            self.processOdomDataset()
            self.ODOM = {'position': self._POSI, 'velocity': self._VELO}
            self.ODOM_LOADED = True
            return self.ODOM
        except Exception as e:
            self.print("[loadOdometry] Unable to interpret, failed. Check variables.\nEnsure: odom_path is a valid string.\nCode: %s" % (e), State.ERROR)
            self.clearOdomVariables()
            return {}
    
    def processOdomDataset(self):
    # Extract from position .csv at path, full odometry
    # Return these as nice lists/dicts

        self.print("[processOdomDataset] Attempting to process odometry from: %s" % (self.ODOM_PATH), State.DEBUG)
        self.TIMES = []
        self._POSI = {'x': [], 'y': [], 'yaw': []}
        self._VELO = {'x': [], 'y': [], 'yaw': []}
        if os.path.isfile(self.ODOM_PATH):
            with open(self.ODOM_PATH, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    self.TIMES.append(          float(row[0]))
                    self._POSI['x'].append(     float(row[1]))
                    self._POSI['y'].append(     float(row[2]))
                    self._POSI['yaw'].append(   float(row[3]))
                    self._VELO['x'].append(     float(row[4]))
                    self._VELO['y'].append(     float(row[5]))
                    self._VELO['yaw'].append(   float(row[6]))
            self.TIMES          = np.array(self.TIMES)       
            self._POSI['x']     = np.array(self._POSI['x'])  
            self._POSI['y']     = np.array(self._POSI['y'])  
            self._POSI['yaw']   = np.array(self._POSI['yaw'])
            self._VELO['x']     = np.array(self._VELO['x'])  
            self._VELO['y']     = np.array(self._VELO['y'])  
            self._VELO['yaw']   = np.array(self._VELO['yaw'])
        else:
            raise Exception("[processOdomDataset] No file at path - cannot continue.")
        return self.TIMES, self._POSI, self._VELO

    def save2npz(self, database_path, filename):
    # Public method. Handles saving the loaded/set attributes of the class instance as a fast loading .npz library/dict file.

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
        # gemerate new full path to save to:
        full_file_path = database_path + separator + filename + "_" + str(self.IMG_DIMS[1]) + ".npz"
        # perform save to compressed numpy file:
        try:
            self.print("[save2npz] Saving data as '%s'." % (full_file_path), State.INFO)
            # Perform actual save operation, where each dict key is assigned as the variable name on the left-hand-side of equality 
            # i.e. filename, ft, x, y, and z will be dict keys, with lists or dicts assigned to them.
            np.savez(full_file_path, img_feats=self.IMG_FEATS, odom=self.ODOM, times=self.TIMES, \
                        odom_path=self.ODOM_PATH, image_paths=self.IMG_PATHS, feat_type=self.FEAT_TYPE, img_dims=self.IMG_DIMS)
        except Exception as e:
            self.print("[save2npz] Unable to perform save operation. Check path.\nCode: %s" % (e), State.ERROR)

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
        # gemerate new full path to save to:\
        file_found = False
        for entry in os.scandir(database_path):
            if entry.is_file() and entry.name.startswith(filename):
                self.print("[_npzLoader] File found: %s" % (entry.name), State.INFO)
                data = np.load(entry.path, allow_pickle=True)
                file_found = True
                break
        if not file_found:
            raise Exception("[_npzLoader] No file found starting with %s in directory '%s'" % (filename, database_path))

        # update class attributes:
        self.IMG_FEATS = data['img_feats'].item()
        self.ODOM = data['odom'].item()
        self.TIMES = data['times']

        self.ODOM_PATH = data['odom_path']
        self.IMG_PATHS = data['image_paths']

        self.FEAT_TYPE = data['feat_type']
        self.IMG_DIMS = tuple(data['img_dims'])

        self.ODOM_LOADED = True
        self.IMAGES_LOADED = True

        self.buildFullDictionary()
        dict_keys = str(np.fromiter(self.SET_DICT.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
        imgs_keys = str(np.fromiter(self.IMG_FEATS.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
        odom_keys = str(np.fromiter(self.ODOM.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
        return dict_keys, imgs_keys, odom_keys

    def npzLoader(self, database_path, filename):
    # Public method. Specify directory path containing .npz file (database_path) and the filename (or an incomplete but unique file prefix)
        try:
            dict_keys, imgs_keys, odom_keys = self._npzLoader(database_path, filename)
            self.print("[npzLoader] Success. Data found with keys: \n\t%s" % (dict_keys), State.INFO)
            self.print("[npzLoader] Image dictionary contains keys: \n\t%s" % (imgs_keys), State.INFO)
            self.print("[npzLoader] Odometry dictionary contains keys: \n\t%s" % (odom_keys), State.INFO)
            return True
        except Exception as e:
            self.print("[npzLoader] Load failed. Check path and file name is correct.\nCode: %s" % (e), State.ERROR)
            return {}

    def npzDatabaseLoadSave(self, database_path, filename, img_paths, odom_path, feat_type, img_dims, do_save=False):
    # Public method. Handles fast loading and slow loading (in the latter, loaded data can be saved for fast loading next time)
    # database_path, filename as per npzLoader.
    # img_paths, odom_path, feat_type, and img_dims as per loadFull
    # do_save=True enables saving for fast loading (with directory/filename specified by database_path and filename respectively)

        if not self.npzLoader(database_path, filename + "_%d" % (img_dims[0])):
            self.print("[npzDatabaseLoadSave] Fast load failed. Building normally.", State.WARN)
            self.SET_DICT = self.loadFull(img_paths, odom_path, feat_type, img_dims)
            if not len(self.SET_DICT):
                raise Exception("[npzDatabaseLoadSave] Normal load failed, fatal error.")
        
            if do_save:
                self.print("[npzDatabaseLoadSave] Build success.", State.INFO)
                self.save2npz(database_path, filename)

        return self.SET_DICT
    

    def clearImageVariables(self):
        self.IMG_PATHS          = ""
        self.FEAT_TYPE          = FeatureType.NONE
        self.IMG_DIMS           = (0,0)
        self.IMG_FEATS          = []
        self.IMAGES_LOADED      = False

    def clearOdomVariables(self):
        self.ODOM_PATH          = ""
        self.odom_x             = []
        self.odom_y             = []
        self.odom_z             = []
        self.ODOM_LOADED        = False

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

### Example usage:

# import rospkg
# import matplotlib.pyplot as plt

# PACKAGE_NAME    = 'aarapsi_intro_pack'
# SET_NAME        = "s2_cw_z_230222_o0_e1"
# FEAT_TYPE       = FeatureType.RAW # Feature Type
# IMG_DIMS        = (64, 64)
# REF_ROOT        = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME
# REF_IMG_PATHS   = [REF_ROOT + "/forward", REF_ROOT + "/left", REF_ROOT + "/right", REF_ROOT + "/panorama"]
# REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odometry.csv"
# DATABASE_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"

# test = VPRImageProcessor()
# dicts = []
# sizes = [8, 16, 32, 64, 400]
# for i in sizes:
#     dicts.append(test.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME + "_%d" % (i), REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, (i,i), do_save=True))
# print((tuple(dicts[0]['img_dims']), (8,8)))
# index = 4
# img = np.reshape(np.array(dicts[index]['img_feats']['panorama'])[0,:],(sizes[index],-1))
# plt.imshow(img)
# plt.show()

# dict_single = test.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME + "_%d" % (8), REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, (8,8), do_save=True)

# print(type((dict_single['odom']['position']['x'])[0]))
# print(type((dict_single['odom']['position']['y'])[0]))
# print(type((dict_single['odom']['position']['yaw'])[0]))
# print(type((dict_single['odom']['velocity']['x'])[0]))
# print(type((dict_single['odom']['velocity']['y'])[0]))
# print(type((dict_single['odom']['velocity']['yaw'])[0]))
# print(type((dict_single['times'])[0]))