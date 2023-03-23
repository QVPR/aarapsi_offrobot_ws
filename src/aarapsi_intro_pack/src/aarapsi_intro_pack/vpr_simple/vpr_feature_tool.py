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
from aarapsi_intro_pack.core.enum_tools import enum_name

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

    def buildFullDictionary(self, dict_in=None):
        if not (dict_in is None):
            self.IMG_FEATS = dict_in['img_feats']
            self.ODOM = dict_in['odom']
            self.TIMES = dict_in['times']
            self.ODOM_PATH = dict_in['odom_path']
            self.IMG_PATHS = dict_in['image_paths']
            self.IMG_DIMS = dict_in['img_dims']
        self.SET_DICT = {'img_feats': self.IMG_FEATS, \
                         'odom': self.ODOM, \
                         'times': self.TIMES,  \
                         'odom_path': self.ODOM_PATH, \
                         'image_paths': self.IMG_PATHS,  \
                         'img_dims': self.IMG_DIMS
                        }
    
    def loadFull(self, img_paths, odom_path, feat_type, img_dims):
    # Load in odometry and image library from raw files (.png and .csv)
    # Feed img_paths to directory containing directories of images
    # odom_path should always be full file path of an odometry.csv file
    # feat_type of enum type FeatureType or a list of enums of type FeatureType
    # img_dims a two-integer positive tuple containing image width and height (width, height) in pixels
    # Returns full dictionary; empty on fail.

        self.print("[loadFull] Attempting to load library.", State.DEBUG)
        if not self.loadImageFeatures(img_paths, feat_type, img_dims): raise Exception("Fatal")
        if not len(self.loadOdometry(odom_path)): raise Exception("Fatal")
        self.buildFullDictionary()
        if not (self.IMAGES_LOADED and self.ODOM_LOADED):
            self.print("[loadFull] Terminating; load procedure failed.", State.FATAL)
            sys.exit()
        return self.SET_DICT

    def loadImageFeatures(self, img_paths, feat_type, img_dims):
    # Load in images

        if not isinstance(img_paths, list):
            if isinstance(img_paths, str):
                self.print("[loadImageFeatures] img_paths is of type str and not type list. Wrapping in a list to proceed.")
                img_paths = [img_paths]
            else:
                raise Exception("img_paths provided is not of type list.")

        if isinstance(feat_type, list):
            if not len(feat_type):
                raise Exception("feat_type provided is an empty list.")
            if not all(isinstance(i, FeatureType) for i in feat_type):
                raise Exception("feat_type provided is a list, but contains elements which are not of type FeatureType.")
            if not (len(feat_type) == len(img_paths)):
                raise Exception("feat_type is of type list yet does not have the same number of elements as img_paths.")
        elif not isinstance(feat_type, FeatureType):
            raise Exception("feat_type provided is not of type FeatureType.")
        else:
            feat_type = [feat_type]
        
        self.IMG_DIMS = img_dims
        self.IMG_PATHS = img_paths

        try:
            self.IMG_FEATS = {}
            for fttype in feat_type:
                self.IMG_FEATS[enum_name(fttype)]  = {}
                for img_path in self.IMG_PATHS:
                    img_set_name = os.path.basename(img_path)
                    self._IMG_DIR = img_path
                    self.print("[loadImageFeatures] Loading set [%s] %s" % (enum_name(fttype), str(img_path)), State.INFO)
                    self.processImageDataset(fttype)
                    self.IMG_FEATS[enum_name(fttype)][img_set_name] = self._IMG_FEATS
            self.IMAGES_LOADED = True
            return len(self.IMG_PATHS)
    
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_paths is a valid string array, feat_type is a valid FeatureType value (not NONE), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.clearImageVariables()
            return 0

    def processImageDataset(self, fttype): 
    # Extract images and their features from path
    # Store in array and return them.

        self.print("[processImageDataset] Attempting to process images from: %s" % (self._IMG_DIR), State.DEBUG)
        imPath_list = np.sort(os.listdir(self._IMG_DIR))
        imPath_list = [os.path.join(self._IMG_DIR, f) for f in imPath_list]

        self._IMG_FEATS = []

        if len(imPath_list) > 0:
            for i, imPath in tqdm(enumerate(imPath_list)):
                frame = cv2.imread(imPath)[:, :, ::-1]
                feat = self.getFeat(frame, fttype=fttype) # fttype: 'downsampled_patchNorm' or 'downsampled_raw'
                self._IMG_FEATS.append(feat)
        else:
            raise Exception("[processImageDataset] No files at path - cannot continue.")
    
    def getFeat(self, im, fttype, dims=None):
    # Get features from im, using VPRImageProcessor's set image dimensions.
    # Specify type via fttype= (from FeatureType enum)
    # Can override the dimensions with dims= (two-element positive integer tuple)
    # Returns feature array, as a flattened array

        if dims is None: # should be almost always unless testing an override
            if not (self.IMG_DIMS[0] > 0 and self.IMG_DIMS[1] > 0):
                raise Exception("[getFeat] Image dimension not set!")
            else:
                dims = self.IMG_DIMS
        if fttype == FeatureType.NONE:
            raise Exception("[getFeat] Feature type not set!")
        try:
            imr = cv2.resize(im, dims)
            ft = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)
            if fttype == FeatureType.RAW:
                pass # already done
            elif fttype == FeatureType.PATCHNORM:
                ft = self.patchNormaliseImage(ft, 8)
            ft_ready = ft.flatten() # np 1d matrix format
            return ft_ready
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))

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
                        odom_path=self.ODOM_PATH, image_paths=self.IMG_PATHS, img_dims=self.IMG_DIMS)
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
        self.IMG_DIMS           = (0,0)
        self.IMG_FEATS          = []
        self.IMAGES_LOADED      = False

    def clearOdomVariables(self):
        self.ODOM_PATH          = ""
        self.odom_x             = []
        self.odom_y             = []
        self.odom_z             = []
        self.ODOM_LOADED        = False

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
        new_spatial_matrix = np.transpose(np.stack(copy.deepcopy([new_spatial_vec[key] for key in list(new_spatial_vec)])))
        groupings = []
        for arr in np.unique(new_spatial_matrix, axis=0): # for each unique row combination:
            groupings.append(list(np.array(np.where(np.all(new_spatial_matrix==arr,axis=1))).flatten())) # indices
            #groupings.append(list(np.array((np.all(new_spatial_matrix==arr,axis=1))).flatten())) # bools
        return new_spatial_vec, groupings
    
    def keep_operation(self, d_in, groupings, mode='first'):
    # Note: order of groupings within the original set can be lost depending on how groupings were generated
        
        # create 'table' where all rows are the same length with first entry as the 'label' (tuple)
        dict_to_list = []
        for bigkey in ['odom', 'img_feats']:
            for midkey in set(d_in[bigkey].keys()):
                for lowkey in set(d_in[bigkey][midkey].keys()):
                    base = [(bigkey, midkey, lowkey)]
                    if bigkey == 'img_feats':
                        obj_raw = np.stack(d_in[bigkey][midkey][lowkey], axis=0)
                        if not (type(np.array(d_in[bigkey][midkey][lowkey]).flatten()[0]) == type(np.uint8)):
                            obj_norm = (obj_raw.astype(np.float64) - np.min(obj_raw)) / (np.max(obj_raw) - np.min(obj_raw))
                            obj_uint8 = np.array(obj_norm * 255, dtype=int)
                            base.extend(obj_uint8)
                        else:
                            base.extend(obj_raw)
                    else:
                        base.extend(d_in[bigkey][midkey][lowkey])
                    dict_to_list.append(base)
        times = [('times',)]
        times.extend(d_in['times'])
        dict_to_list.append(times)
        np_dict_to_list = np.transpose(np.array(dict_to_list, dtype=object))

        # extract rows
        groups_store = []
        for group in groupings:
            if len(group) < 1:
                continue
            if mode=='average':
                groups_store.append(np.mean(np_dict_to_list[1:, :][group,:], axis=0))
                continue
            elif mode=='first': 
                index = 0
            elif mode=='random': 
                index = int(np.random.rand() * (len(group) - 1))
            # for first and random modes:
            index_to_keep = group[index] + 1 # +1 accounts for label
            groups_store.append(np_dict_to_list[index_to_keep, :])

        # restructure and reorder
        cropped_store = np.array(groups_store)
        ind = -1
        for c, i in enumerate(np_dict_to_list[0,:]):
            if i[0] == 'times':
                ind = c
                break
        if ind == -1: raise Exception("Fatal")
        cropped_reorder = cropped_store[cropped_store[:,-1].argsort()]
        d_in.pop('times')
        d_in['times'] = cropped_reorder[:,c]

        # convert back to dictionary and update old dictionary entries
        for bigkey in ['odom', 'img_feats']:
            for midkey in set(d_in[bigkey].keys()):
                for lowkey in set(d_in[bigkey][midkey].keys()):
                    for c, i in enumerate(np_dict_to_list[0,:]):
                        if (bigkey,midkey,lowkey) == i:
                            d_in[bigkey][midkey].pop(lowkey)
                            if bigkey == 'img_feats':
                                d_in[bigkey][midkey][i[2]] = np.stack(copy.deepcopy(cropped_reorder[:,c]),axis=0).astype(np.uint8)
                            else:
                                d_in[bigkey][midkey][i[2]] = copy.deepcopy(cropped_reorder[:,c])
        

        return d_in
    
    def filter(self, metrics=None, mode=None, keep='first'):
        if not len(self.SET_DICT):
            raise Exception("[filter] Full dictionary not yet built.")
        filtered = copy.deepcopy(self.SET_DICT) # ensure we don't change the original dictionary

        if mode is None:
            return filtered
        valid_keeps = ['first', 'random', 'average']
        if not keep in valid_keeps: # ensure valid
            raise Exception('[filter] Unsupported keep style %s. Valid keep styles: %s' % (str(keep), str(valid_keeps)))
        valid_modes = ['position', 'velocity']
        if not mode in valid_modes: # ensure valid
            raise Exception('[filter] Unsupported mode %s. Valid modes: %s' % (str(mode), str(valid_modes)))
        
        # Perform filter step:
        if mode in ['position', 'velocity']: # valid inputs to roundSpatial()
            (filtered['odom'][mode], groupings) = self.roundSpatial(filtered['odom'][mode], metrics)
        #elif mode in []: #TODO
        #    pass
        filtered = self.keep_operation(filtered, groupings, keep) # remove duplications
        return filtered

### Example usage:
if __name__ == '__main__':
    import rospkg
    from aarapsi_intro_pack.core.helper_tools import vis_dict

    PACKAGE_NAME        = 'aarapsi_intro_pack'

    FEAT_TYPE           = [FeatureType.RAW, FeatureType.PATCHNORM] # Feature Type
    REF_ROOT            = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/"
    DATABASE_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"
    DATABASE_PATH_FILT  = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/filt"

    SET_NAMES           = [ 's1_ccw_o0_e0_a0', 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0',\
                            's2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
    sizes               = [ 8, 16, 32, 64, 128, 400]

    REF_IMG_PATHS       = [ REF_ROOT + SET_NAMES[0] + "/forward", \
                            #REF_ROOT + SET_NAMES[0] + "/left", \
                            #REF_ROOT + SET_NAMES[0] + "/right", \
                            REF_ROOT + SET_NAMES[0] + "/panorama"]
    REF_ODOM_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAMES[0] + "/odometry.csv"

    ip = VPRImageProcessor() # reinit just to clean house
    IMG_DIMS = (sizes[3],)*2
    ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAMES[0], REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)
    prefilt = copy.deepcopy(ip.SET_DICT)
    filtered = ip.filter(keep='average', mode='position', metrics={'x': 0.1, 'y': 0.1, 'yaw': (10*2*np.pi/360)})
    ip.buildFullDictionary(dict_in=filtered)
    vis_dict(prefilt)
    vis_dict(filtered)
    #Visualise discretisation:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1,1)
    #axes.plot(np.arange(len(filtered['times'])), filtered['times'], '.')
    axes.plot(filtered['odom']['position']['x'], filtered['odom']['position']['y'], '.', markersize=2)
    axes.plot(prefilt['odom']['position']['x'], prefilt['odom']['position']['y'], '.', markersize=2)
    axes.set_aspect('equal', 'box')
    #plt.show()

    # Visualise forward camera feed:

    def prep_for_video(img):
        _min = np.min(img)
        _max = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        _img_uint = np.array(_img_norm * 255, dtype=np.uint8)
        img_processed = np.stack(copy.deepcopy(_img_uint), axis=0)
        return img_processed

    fps = 40.0
    data_for_vid = filtered # prefilt, filtered
    video_norm = cv2.VideoWriter('norm_feed.avi', 0, fps, IMG_DIMS)
    for i in range(len(data_for_vid['times'])):
        video_norm.write(np.dstack((np.reshape(prep_for_video(data_for_vid['img_feats']['PATCHNORM']['forward'][i]), IMG_DIMS),)*3))
    video_norm.release()
    video_raw = cv2.VideoWriter('raw_feed.avi', 0, fps, IMG_DIMS)
    for i in range(len(data_for_vid['times'])):
        video_raw.write(np.dstack((np.reshape(prep_for_video(data_for_vid['img_feats']['RAW']['forward'][i]), IMG_DIMS),)*3))
    video_raw.release()

    # Export single image:
    img = np.dstack((np.reshape(filtered['img_feats']['RAW']['forward'][0], IMG_DIMS),)*3)
    cv2.imwrite('test.png', img)

    # # Iterate through multiple:

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
            
    #         # create filtered database:
    #         filtered = ip.filter(keep='first', mode='position', metrics={'x': 0.1, 'y': 0.1, 'yaw': (10*2*np.pi/360)})
    #         ip.buildFullDictionary(dict_in=filtered)
    #         ip.save2npz(DATABASE_PATH_FILT, SET_NAME)

