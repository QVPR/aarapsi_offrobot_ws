#!/usr/bin/env python3
import numpy as np
import os
import cv2
from enum import Enum
from tqdm import tqdm

class FeatureType(Enum):
    NONE = 0
    RAW = 1
    PATCHNORM = 2

class VPRImageProcessor: # main ROS class
    def __init__(self):

        self.clearImageVariables()
        self.clearOdomVariables()

    def loadFull(self, img_path, odom_path, feat_type, img_dims):
        self.loadImageFeatures(img_path, feat_type, img_dims)
        self.loadOdometry(odom_path)
        return self._img_info, self._odom

    def loadImageFeatures(self, img_path, feat_type, img_dims):
        self.IMG_PATH = img_path
        self.FEAT_TYPE = feat_type
        self.IMG_DIMS = img_dims
        try:
            self.processImageDataset()
            self.image_features = list(self._img_info['fts'])
            if not self.image_features: # if empty
                raise Exception("Output is empty. Check inputs.")
            self.IMAGES_LOADED = True
            return self.image_features
        except Exception as e:
            print("[ERROR][loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_path is a valid string, feat_type is a valid FeatureType value (not NONE!), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nError: %s" % (e))
            self.clearImageVariables()
            return []
    
    def loadOdometry(self, odom_path):
        self.ODOM_PATH = odom_path
        try:
            self.processOdomDataset()
            self.image_odom_x = list(self._odom['x'])
            self.image_odom_y = list(self._odom['y'])
            self.image_odom_z = list(self._odom['z'])
            if not (self.image_odom_x and self.image_odom_y): # if empty
                raise Exception("Output is empty. Check inputs.")
            if not (self.image_odom_z): # may not exist
                print("[WARN][loadOdometry] Odometry has no z values.")
            self.ODOM_LOADED = True
            return self.image_odom_x, self.image_odom_y, self.image_odom_z
        except Exception as e:
            print("[ERROR][loadOdometry] Unable to interpret, failed. Check variables.\nEnsure: odom_path is a valid string.\nError: %s" % (e))
            self.clearOdomVariables()
            return [], [], []

    def npzLoader(self, filename):
        try:
            # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
            ext = os.path.splitext(filename)[-1].lower()
            if ext == "":
                filename = filename + '.npz'
            elif not (ext == ".npz"):
                raise Exception("File is not of type .npz!")
            data = np.load(filename)
            print("[npzLoader] Success. Data found with keys: " + str(np.fromiter(data.keys(), (str, 10))))
            return data
        except Exception as e:
            print("[ERROR][npzLoader] Load failed. Check path and file name is correct.\nError: %s" % (e))
    
    def save2npz(self, filename):
        if not (self.IMAGES_LOADED):
            raise Exception("No images loaded. loadImageFeatures() must be performed before any save process can be performed.")
        if not (self.ODOM_LOADED):
            print("[WARN][save2npz] No odometry loaded: rows will be empty.")
        try:
            np.savez(filename, ft=self.image_features, x=self.image_odom_x, y=self.image_odom_y, z=self.image_odom_z, odompath=self.ODOM_PATH, imgpath=self.IMG_PATH, fttype=self.FEAT_TYPE, imgdims=self.IMG_DIMS)
        except Exception as e:
            print("[ERROR][save2npz] Unable to perform save operation. Check path.\nError: %s" % (e))

    def clearImageVariables(self):
        self.IMG_PATH           = ""
        self.FEAT_TYPE          = FeatureType.NONE
        self.IMG_DIMS           = (0,0)
        self.image_features     = []
        self.IMAGES_LOADED      = False

    def clearOdomVariables(self):
        self.ODOM_PATH      = ""
        self.image_odom_x   = []
        self.image_odom_y   = []
        self.image_odom_z   = []
        self.ODOM_LOADED    = False

    def processImageDataset(self): 
    # Extract images and their features from path
    # Store in arrays and return them.

        imPath_list = np.sort(os.listdir(self.IMG_PATH))
        imPath_list = [os.path.join(self.IMG_PATH, f) for f in imPath_list]

        self._img_info = {'paths': imPath_list, 'fts': []}

        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame) # ftType: 'downsampled_patchNorm' or 'downsampled_raw'
                self._img_info['fts'].append(feat)
        else:
            raise Exception("No files at path - cannot continue.")

    def getFeat(self, im, size=1, dims=None, fttype=None):
    # Get features from im, using VPRImageProcessor's set image dimensions and feature type (from loadImageFeatures).
    # Can override the dimensions and feature type using fttype= (from FeatureType enum) and dims= (two-element positive integer tuple)
    # Returns feature arary, as a flattened array (size=1) or a flattened array reshaped to 2d matrix format (size=2).

        if dims is None:
            if self.IMG_DIMS == (0,0):
                raise Exception("[ERROR][getFeat] Image dimension not set!")
            else:
                dims = self.IMG_DIMS
        if fttype is None:
            if self.FEAT_TYPE == FeatureType.NONE:
                raise Exception("[ERROR][getFeat] Feature type not set!")
            else:
                fttype = self.FEAT_TYPE
        if not ( isinstance(size, int) and (size in [1, 2]) ):
            raise Exception("[ERROR][getFeat] Size must be either integer 1 or 2.")
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
            raise Exception("[ERROR][getFeat] Feature vector could not be constructed.\nError: %s" % (e))

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
            raise Exception("No files at path - cannot continue.")

### Example usage:

# import rospkg

# PACKAGE_NAME    = 'aarapsi_intro_pack'
# REF_IMG_PATH    = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/ccw_loop/forward"
# REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/ccw_loop/odo"
# SAVE_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/test.npz"
# FEAT_TYPE       = FeatureType.RAW # Feature Type
# IMG_DIMS        = (32, 32)

# test = VPRImageProcessor()
# test.loadImageFeatures(REF_IMG_PATH, FEAT_TYPE, IMG_DIMS)
# test.loadOdometry(REF_ODOM_PATH)
# test.save2npz(SAVE_PATH)
# data = test.npzLoader(SAVE_PATH)