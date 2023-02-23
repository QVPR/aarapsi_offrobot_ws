#!/usr/bin/env python3

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

import numpy as np
import math
import cv2

import matplotlib.pyplot as plt

class mrc:
    def __init__(self, node_name='lidar2panorama', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        lidar_topic = '/velodyne_points'

        lidar_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.lidar_callback, queue_size=1)

        ## Parse all the inputs:
        self.rate_num               = rate_num # Hz
        self.rate_obj               = rospy.Rate(self.rate_num)

        self.lidar_msg              = PointCloud2()
        self.new_lidar_msg          = False

        # https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Panorama-view.html
        self.SURROUND_U_STEP = 1.    #resolution
        self.SURROUND_V_STEP = 1.33
        self.SURROUND_U_MIN, self.SURROUND_U_MAX = np.array([0,    360])/self.SURROUND_U_STEP  # horizontal of cylindrial projection
        self.SURROUND_V_MIN, self.SURROUND_V_MAX = np.array([-90,   90])/self.SURROUND_V_STEP  # vertical   of cylindrial projection

    def lidar_callback(self, msg):
        self.lidar_msg              = msg
        self.new_lidar_msg          = True

    def lidar_to_surround_coords(self, x, y, z, d ):
        # https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Panorama-view.html
        u =   np.arctan2(x, y)/np.pi*180 /self.SURROUND_U_STEP
        v = - np.arctan2(z, d)/np.pi*180 /self.SURROUND_V_STEP
        u = (u +90)%360  ##<todo> car will be spit into 2 at boundary  ...

        u = np.rint(u)
        v = np.rint(v)
        u = (u - self.SURROUND_U_MIN).astype(np.uint8)
        v = (v - self.SURROUND_V_MIN).astype(np.uint8)

        return u,v


    def main_loop(self, imshow_handle):
        def normalise_to_255(a):
            return (((a - min(a)) / float(max(a) - min(a))) * 255).astype(np.uint8)

        if not self.new_lidar_msg:
            return
        
        pc_xyzi = np.array(list(point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))#, "ring", "time"
        #d = np.sqrt(pc_xyzi[:,0]**2 + pc_xyzi[:,1]**2) # distance relative to origin ignoring 'z'
        #t = np.arctan2(pc_xyzi[:,1], pc_xyzi[:,0])

        # plt.sca(axes)
        # plt.cla()
        # axes.plot(t, pc_xyzi[:,2])
        # axes.set(xlabel='Angle (rad)', ylabel='Z-Axis')
        
        # https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Panorama-view.html
        x = pc_xyzi[:,0]
        y = pc_xyzi[:,1]
        z = pc_xyzi[:,2]
        r = pc_xyzi[:,3]

        d = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin
        u,v = self.lidar_to_surround_coords(x,y,z,d)

        width  = int(self.SURROUND_U_MAX - self.SURROUND_U_MIN + 1)
        height = int(self.SURROUND_V_MAX - self.SURROUND_V_MIN + 1)
        surround     = np.zeros((height, width, 3), dtype=np.float32)
        surround_img = np.zeros((height, width, 3), dtype=np.uint8)

        surround[v, u, 0] = d
        surround[v, u, 1] = z
        surround[v, u, 2] = r
        surround_img[v, u, 0] = normalise_to_255(np.clip(d,     0, 30))
        surround_img[v, u, 1] = normalise_to_255(np.clip(z+1.8, 0, 100))
        surround_img[v, u, 2] = normalise_to_255(np.clip(r,     0, 30))

        surround_img_resized = cv2.resize(surround, (100, 400), interpolation = cv2.INTER_AREA)
        imshow_handle.set_data(surround_img_resized)
        #imshow_handle.autoscale()
        
        

if __name__ == '__main__':
    try:
        nmrc = mrc()

        fig, axes = plt.subplots(1, 1, figsize=(15,4))
        imshow_handle = axes.imshow(np.zeros((100,400,3)))
        fig.show()

        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            nmrc.main_loop(imshow_handle)
            fig.canvas.draw()
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
