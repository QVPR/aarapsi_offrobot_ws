#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import rospkg

class mrc: # main ROS class
    def __init__(self):
        self.robot_x    = 0
        self.robot_y    = 0
        self.robot_z    = 0
        self.frame_n    = 0

        rospy.init_node('odom_image_processor', anonymous=True)
        self.rate_num   = 5 # Hz
        self.rate_obj   = rospy.Rate(1/self.rate_num)

        self.odom_sub   = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        self.img0_sub   = rospy.Subscriber("/ros_indigosdk_occam/image0", Image, self.img0_callback)

        self.store_img0 = Image()

        self.new_img0   = False

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.robot_x = round(msg.pose.pose.position.x, 3)
        self.robot_y = round(msg.pose.pose.position.y, 3)
        self.robot_z = round(msg.pose.pose.position.z, 3)

    def img0_callback(self, msg):
    # /ros_indigosdk_occam/image0 (sensor_msgs/Image)
    # Store newest image received

        self.store_img0 = msg
        self.new_img0 = True

def odom_image_processor():
    nmrc    = mrc() # make new class instance
    bridge  = CvBridge() # to convert sensor_msgs/Image to cv2

    rospack = rospkg.RosPack()

    path_for_dataset = rospack.get_path('aarapsi_intro_pack') + '/data/set_1'

    while not rospy.is_shutdown():

        nmrc.rate.sleep()

        if not nmrc.new_img0:
            continue

        rospy.loginfo("Logging frame {}".format(str(nmrc.frame_n).zfill(6)))
        image0 = bridge.imgmsg_to_cv2(nmrc.store_img0, "bgr8")
        cv.imwrite(path_for_dataset + '/images/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6)), image0)
        np.savetxt(path_for_dataset + '/odo/frame_id_{}.csv'.format(str(nmrc.frame_n).zfill(6)), np.array([nmrc.robot_x, nmrc.robot_y, nmrc.robot_z]),delimiter=',')
        nmrc.frame_n += 1

        nmrc.new_img0 = False

        

    rospy.loginfo("Exitting node (roscore condition termination).")

if __name__ == '__main__':
    try:
        odom_image_processor()
    except rospy.ROSInterruptException:
        pass