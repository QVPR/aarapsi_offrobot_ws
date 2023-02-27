#!/usr/bin/env python3

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

import numpy as np
import math
import cv2
import sys
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

    def lidar_callback(self, msg):
        self.lidar_msg              = msg
        self.new_lidar_msg          = True

    def main_loop(self, imsh, tdva, plra, pix_x, pix_y):

        if not self.new_lidar_msg:
            return

        # Hacked for ROS + our LiDAR from:
        # https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Panorama-view.html
        def normalise_to_255(a):
            return (((a - min(a)) / float(max(a) - min(a))) * 255).astype(np.uint8)

        def lidar_to_surround_coords(x, y, z, d ):
            u =   np.arctan2(x, y) *(180/np.pi)
            v = - np.arctan2(z, d) *(180/np.pi)
            u = (u +90)%360  ##<todo> car will be spit into 2 at boundary  ...

            u = np.rint(u).astype(int)
            v = np.rint(v-min(v)).astype(int)

            return u,v
        
        pc_xyzi = np.array(list(point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))#, "ring", "time"
        pc_xyzi = np.delete(pc_xyzi, np.arange(pc_xyzi.shape[0])[pc_xyzi[:,2] < -0.5], 1)
        x = np.array(pc_xyzi[:,0])
        y = np.array(pc_xyzi[:,1])
        z = np.array(pc_xyzi[:,2])
        r = np.array(pc_xyzi[:,3]) # intensity, reflectance

        d = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin
        u,v = lidar_to_surround_coords(x,y,z,d)

        width                   = 361
        height                  = 31
        panorama                = np.zeros((height, width, 3), dtype=np.uint8)
        panorama[v, u, 1]       = normalise_to_255(d)
        panorama[v, u, 2]       = normalise_to_255(z)
        panorama[v, u, 0]       = normalise_to_255(r)

        image = cv2.resize(panorama[::2], (pix_x, pix_y), interpolation = cv2.INTER_AREA)
        imsh.set_data(image)
        tdva.cla()
        tdva.scatter(x, y, c=z)
        tdva.set_aspect('equal')
        tdva.set_xlim(-15, 15)
        tdva.set_ylim(-15, 15)

        # https://kaleidoscopicdiaries.wordpress.com/2015/05/16/planet-panoramas-in-python/
        panorama_ud = np.flipud(panorama).astype(int)
        rr = np.linspace(0.01, 1, panorama_ud.shape[0])
        th = np.linspace(0, 2*np.pi, panorama_ud.shape[1])
        Rr, Th = np.meshgrid(rr, th)
        pano_ud_trans = np.transpose(-panorama_ud)
        plra.cla()
        plra.pcolor(Th, Rr, pano_ud_trans)

if __name__ == '__main__':
    try:
        nmrc = mrc()

        fig, axes = plt.subplots(2, 2, figsize=(15,8))
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
        gs = axes[0, 0].get_gridspec()
        for ax in axes[0,:]:
            print(ax)
            ax.remove()
        axbig = fig.add_subplot(gs[0,:])
        pix_x = 361
        pix_y = 91

        image = np.zeros((pix_y, pix_x, 3))
        imsh = axbig.imshow(image, cmap='gist_rainbow', vmin=0, vmax=1)
        
        fig.show()

        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            nmrc.main_loop(imsh, axes[1,0], axes[1,1], pix_x, pix_y)
            fig.canvas.draw()
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
