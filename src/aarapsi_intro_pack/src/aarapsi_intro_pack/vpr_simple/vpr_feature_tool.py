#!/usr/bin/env python3

import numpy as np
import copy
try:
    import rospy # used for printing
except:
    pass
import os
import cv2
import sys
import csv
from enum import Enum
from tqdm import tqdm
from aarapsi_intro_pack.core.helper_tools import combine_dicts

# For image processing type
class FeatureType(Enum):
    NONE = 0
    RAW = 1
    PATCHNORM = 2
    NETVLAD = 3
    HYBRID  = 4

# For logging
class State(Enum):
    DEBUG = "[DEBUG]"
    INFO = "[INFO]"
    WARN = "[WARN]"
    ERROR = "[!ERROR!]"
    FATAL = "[!!FATAL!!]"

class VPRImageProcessor: # main ROS class
    def __init__(self, ros=False):

        self.clearImageVariables()
        self.clearOdomVariables()
        self.ros = ros

    def print(self, text, state):
    # Print function helper
    # For use with integration with ROS
        try:
            if self.ros: # if used inside of a running ROS node
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
                raise Exception
        except:
            print(state.value + " " + str(text))

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
        # gemerate new full path to save to:
        filename_extended = database_path + separator + filename + "_%dx" % (self.IMG_DIMS[0])
        if not (self.IMG_DIMS[0] == self.IMG_DIMS[1]):
            filename_extended = filename_extended + "%d" % (self.IMG_DIMS[1])
        full_file_path = filename_extended + ".npz"
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
        # gemerate new full path to save to:
        file_found = False
        for entry in os.scandir(database_path):
            if entry.is_file() and entry.name.startswith(filename):
                self.print("[_npzLoader] File found: %s" % (entry.name), State.INFO)
                data = np.load(entry.path, allow_pickle=True)
                file_found = True
                break
        if not file_found:
            raise Exception("[_npzLoader] No file found starting with %s in directory '%s'" % (filename, database_path))

        try:
            self.print("[_npzLoader] Data Main Keys: %s" % (str(np.fromiter(data.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            self.print("[_npzLoader] Data Image Keys: %s" % (str(np.fromiter(data['img_feats'].item().keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            self.print("[_npzLoader] Data Odom Keys: %s" % (str(np.fromiter(data['odom'].item().keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            # update class attributes:
            self.IMG_FEATS = data['img_feats'].item()
            self.ODOM = data['odom'].item()
            self.TIMES = data['times']

            self.ODOM_PATH = data['odom_path']
            self.IMG_PATHS = data['image_paths']

            self.FEAT_TYPE = data['feat_type'].item()
            self.IMG_DIMS = tuple(data['img_dims'])

            self.ODOM_LOADED = True
            self.IMAGES_LOADED = True

            self.buildFullDictionary()
            dict_keys = str(np.fromiter(self.SET_DICT.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            imgs_keys = str(np.fromiter(self.IMG_FEATS.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            odom_keys = str(np.fromiter(self.ODOM.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            return dict_keys, imgs_keys, odom_keys
        except Exception as e:
            raise Exception("[_npzLoader] Error: %s" % (e))

    def npzLoader(self, database_path, filename, img_dims):
    # Public method. Specify directory path containing .npz file (database_path) and the filename (or an incomplete but unique file prefix)
        try:

            while 'x' in filename.split('_')[-1]: # clean image size off end
                filename = filename[0:-1-len(filename.split('_')[-1])]
            filename_extended = filename + "_%dx" % (img_dims[0])
            if not (self.IMG_DIMS[0] == self.IMG_DIMS[1]):
                filename_extended = filename_extended + "%d" % (img_dims[1])

            dict_keys, imgs_keys, odom_keys = self._npzLoader(database_path, filename_extended)
            self.print("[npzLoader] Success. Data found with keys: \n\t%s" % (dict_keys), State.DEBUG)
            self.print("[npzLoader] Image dictionary contains keys: \n\t%s" % (imgs_keys), State.DEBUG)
            self.print("[npzLoader] Odometry dictionary contains keys: \n\t%s" % (odom_keys), State.DEBUG)
            return True
        except Exception as e:
            self.print("[npzLoader] Load failed. Check path and file name is correct.\nCode: %s" % (e), State.ERROR)
            return {}

    def npzDatabaseLoadSave(self, database_path, filename, img_paths, odom_path, feat_type, img_dims, do_save=False):
    # Public method. Handles fast loading and slow loading (in the latter, loaded data can be saved for fast loading next time)
    # database_path, filename as per npzLoader.
    # img_paths, odom_path, feat_type, and img_dims as per loadFull
    # do_save=True enables saving for fast loading (with directory/filename specified by database_path and filename respectively)

        if not self.npzLoader(database_path, filename, img_dims):
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

    def getFeat(self, im, dims=None, fttype=None):
    # Get features from im, using VPRImageProcessor's set image dimensions and feature type (from loadImageFeatures).
    # Can override the dimensions and feature type using fttype= (from FeatureType enum) and dims= (two-element positive integer tuple)
    # Returns feature array, as a flattened array

        if dims is None: # should be almost always unless testing an override
            if not (self.IMG_DIMS[0] > 0 and self.IMG_DIMS[1] > 0):
                raise Exception("[getFeat] Image dimension not set!")
            else:
                dims = self.IMG_DIMS
        if fttype is None:
            if self.FEAT_TYPE == FeatureType.NONE:
                raise Exception("[getFeat] Feature type not set!")
            else:
                fttype = self.FEAT_TYPE
        try:
            imr = cv2.resize(im, dims)
            ft = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)
            if self.FEAT_TYPE == FeatureType.RAW:
                pass # already done
            elif self.FEAT_TYPE == FeatureType.PATCHNORM:
                ft = self.patchNormaliseImage(ft, 8)
            ft_ready = ft.flatten() # np 1d matrix format
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

    def roundSpatial(self, spatial_vec, metrics=None):
        if metrics is None:
            metrics = {'x': 0.05, 'y': 0.05, 'yaw': (2*np.pi/360)}
        new_spatial_vec = {}
        for key in metrics:
            new_spatial_vec[key] = np.round(np.array(spatial_vec[key])/metrics[key],0) * metrics[key]
        new_spatial_matrix = np.transpose(np.stack([new_spatial_vec[key] for key in list(new_spatial_vec)]))
        groupings = []
        for arr in np.unique(new_spatial_matrix, axis=0): # for each unique row combination:
            #groupings.append(list(np.array(np.where(np.all(new_spatial_matrix==arr,axis=1))).flatten())) # indices
            groupings.append(list(np.array((np.all(new_spatial_matrix==arr,axis=1))).flatten())) # bools
        return new_spatial_vec, groupings
    
    def keep_first(self, d_in, groupings):
    # Note: order of groupings within the original set can be lost depending on how groupings were generated

        pos_dict_list = []
        vel_dict_list = []
        img_dict_list = []
        for group in groupings:
            d_in['times'][group][1:] = []
            pos_dict_list.append({ key: d_in['odom']['position'][key][group][0]
                                   for key in set(d_in['odom']['position'].keys())})
            vel_dict_list.append({ key: d_in['odom']['velocity'][key][group][0]
                                   for key in set(d_in['odom']['velocity'].keys())})
            img_dict_list.append({ key: d_in['img_feats'][key][group][0]
                                   for key in set(d_in['img_feats'].keys())})
        d_in['odom']['position'] = combine_dicts(pos_dict_list)
        d_in['odom']['velocity'] = combine_dicts(vel_dict_list)
        d_in['img_feats']        = combine_dicts(img_dict_list)
        return d_in
    
    def keep_random(self, d_in, groupings):
    # Note: order of groupings within the original set can be lost depending on how groupings were generated

        pos_dict_list = []
        vel_dict_list = []
        img_dict_list = []
        for group in groupings:
            index_to_keep = int(np.random.rand() * (len(group) - 1)) # pick a random element from every group
            d_in['times'][group][1:] = []
            pos_dict_list.append({ key: d_in['odom']['position'][key][group][index_to_keep] 
                                   for key in set(d_in['odom']['position'].keys())})
            vel_dict_list.append({ key: d_in['odom']['velocity'][key][group][index_to_keep] 
                                   for key in set(d_in['odom']['velocity'].keys())})
            img_dict_list.append({ key: d_in['img_feats'][key][group][index_to_keep] 
                                   for key in set(d_in['img_feats'].keys())})
        d_in['odom']['position'] = combine_dicts(pos_dict_list)
        d_in['odom']['velocity'] = combine_dicts(vel_dict_list)
        d_in['img_feats']        = combine_dicts(img_dict_list)
        return d_in
    
    def keep_average(self, d_in, groupings):
    # Note: order of groupings within the original set can be lost depending on how groupings were generated

        pos_dict_list = []
        vel_dict_list = []
        img_dict_list = []
        for group in groupings:
            d_in['times'][group][1:] = []
            pos_dict_list.append({ key: np.mean(d_in['odom']['position'][key][group])
                                   for key in set(d_in['odom']['position'].keys())})
            vel_dict_list.append({ key: np.mean(d_in['odom']['velocity'][key][group])
                                   for key in set(d_in['odom']['velocity'].keys())})
            img_dict_list.append({ key: np.mean(d_in['img_feats'][key][group], axis=0, dtype=np.uint8)
                                   for key in set(d_in['img_feats'].keys())})
        d_in['odom']['position'] = combine_dicts(pos_dict_list)
        d_in['odom']['velocity'] = combine_dicts(vel_dict_list)
        d_in['img_feats']        = combine_dicts(img_dict_list)
        return d_in
    
    def filter(self, metric=None, mode=None, keep='first'):
        if not len(self.SET_DICT):
            raise Exception("[filter] Full dictionary not yet built.")
        filtered = copy.deepcopy(self.SET_DICT) # ensure we don't change the original dictionary

        if mode is None:
            return filtered

        keep_styles = {'first': self.keep_first, 'random': self.keep_random, 'average': self.keep_average}
        valid_keeps = list(keep_styles.keys())
        if not keep in valid_keeps: # ensure valid
            raise Exception('[filter] Unsupported keep style %s. Valid keep styles: %s' % (str(keep), str(valid_keeps)))
        valid_modes = ['position', 'velocity']
        if not mode in valid_modes: # ensure valid
            raise Exception('[filter] Unsupported mode %s. Valid modes: %s' % (str(mode), str(valid_modes)))
        
        # Perform filter step:
        if mode in ['position', 'velocity']: # valid inputs to roundSpatial()
            (filtered['odom'][mode], groupings) = self.roundSpatial(filtered['odom'][mode], metric)
        #elif mode in []: #TODO
        #    pass
        filtered = keep_styles[keep](filtered, groupings) # remove duplications
        return filtered

### Example usage:

# import rospkg

# PACKAGE_NAME    = 'aarapsi_intro_pack'

# FEAT_TYPE       = FeatureType.RAW # Feature Type
# REF_ROOT        = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/"
# DATABASE_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"

# SET_NAMES       = ['s1_ccw_o0_e0_a0']#, 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0',\
#                    #'s2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
# sizes           = [8, 16, 32, 64, 128, 400]

# for SET_NAME in SET_NAMES:
#     REF_IMG_PATHS   = [ REF_ROOT + SET_NAME + "/forward", \
#                         REF_ROOT + SET_NAME + "/left", \
#                         REF_ROOT + SET_NAME + "/right", \
#                         REF_ROOT + SET_NAME + "/panorama"]
#     REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odometry.csv"

#     for i in sizes:
#         IMG_DIMS = (i, i)
#         ip = VPRImageProcessor() # reinit just to clean house
#         ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME, REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)
