#!/usr/bin/env python3

# import rospkg
from aarapsi_intro_pack import FeatureType, VPRImageProcessor
import rospkg
from aarapsi_intro_pack.core.helper_tools import vis_dict
import numpy as np
import copy
import cv2
import os
import sys

### Example usage:
if __name__ == '__main__':

    PACKAGE_NAME        = 'aarapsi_intro_pack'

    FEAT_TYPE           = [FeatureType.RAW, FeatureType.PATCHNORM] # Feature Type
    REF_ROOT            = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/"
    DATABASE_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"
    DATABASE_PATH_FILT  = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/filt"

    SET_NAMES           = [ 's1_ccw_o0_e0_a0', 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0',\
                            's2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
    sizes               = [ 8, 16, 32, 64, 128, 400]

    REF_IMG_PATHS       = [ REF_ROOT + SET_NAMES[0] + "/forward", ]#\
                            #REF_ROOT + SET_NAMES[0] + "/left", \
                            #REF_ROOT + SET_NAMES[0] + "/right", \
                            #REF_ROOT + SET_NAMES[0] + "/panorama"]
    REF_ODOM_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAMES[0] + "/odometry.csv"

    ip = VPRImageProcessor() # reinit just to clean house
    IMG_DIMS = (sizes[3],)*2
    ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAMES[0], REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)
    print(type(ip.SET_DICT['img_feats']['RAW']['forward'].flatten()[0]))
    print(type(ip.SET_DICT['img_feats']['PATCHNORM']['forward'].flatten()[0]))
    prefilt = copy.deepcopy(ip.SET_DICT)
    filtered = ip.filter(keep='average', mode='position', metrics={'x': 0.1, 'y': 0.1, 'yaw': (10*2*np.pi/360)})
    print(type(filtered['img_feats']['RAW']['forward'].flatten()[0]))
    print(type(filtered['img_feats']['PATCHNORM']['forward'].flatten()[0]))
#     ip.buildFullDictionary(dict_in=filtered)
#     vis_dict(prefilt)
#     vis_dict(filtered)
#     #Visualise discretisation:
#     import matplotlib
#     matplotlib.use('TkAgg')
#     from matplotlib import pyplot as plt

    # fig, axes = plt.subplots(1,1)
    # #axes.plot(np.arange(len(filtered['times'])), filtered['times'], '.')
    # axes.plot(filtered['odom']['position']['x'], filtered['odom']['position']['y'], '.', markersize=2)
    # axes.plot(prefilt['odom']['position']['x'], prefilt['odom']['position']['y'], '.', markersize=2)
    # axes.set_aspect('equal', 'box')
    #plt.show()

    # Visualise forward camera feed:

    def prep_for_video(img):
        _min = np.min(img)
        _max = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        _img_uint = np.array(_img_norm * 255, dtype=np.uint8)
        return _img_uint

    fps = 40.0
    data_for_vid = prefilt # prefilt, filtered
    video_norm = cv2.VideoWriter('norm_feed.avi', 0, fps, IMG_DIMS)
    for i in range(len(data_for_vid['times'])):
        video_norm.write(np.dstack((np.reshape(prep_for_video(data_for_vid['img_feats']['PATCHNORM']['forward'][i]), IMG_DIMS),)*3))
    video_norm.release()
    video_raw = cv2.VideoWriter('raw_feed.avi', 0, fps, IMG_DIMS)
    for i in range(len(data_for_vid['times'])):
        video_raw.write(np.dstack((np.reshape(prep_for_video(data_for_vid['img_feats']['RAW']['forward'][i]), IMG_DIMS),)*3))
    video_raw.release()

    # # Export single image:
    # img = np.dstack((np.reshape(filtered['img_feats']['RAW']['forward'][0], IMG_DIMS),)*3)
    # cv2.imwrite('test.png', img)

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