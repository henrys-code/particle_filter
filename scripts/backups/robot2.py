#!/usr/bin/env python

import rospy
import math as m
import numpy as np
import random
from map_utils import Map
from copy import deepcopy
from read_config import read_config
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray, PointStamped, Quaternion, Point
from sklearn.neighbors import KDTree
from helper_functions import *
class Robot():
    def __init__(self):
        self.config = read_config()
        rospy.init_node('robot')
        self.pose_array = PoseArray()
        self.pose_array.header.stamp = rospy.Time.now()
        self.pose_array.header.frame_id = 'map'
        self.pose_array.poses = []
        self.moves = self.config["move_list"]
        #Particle update values to use
        self.isFirstMove = True
        self.first_sigma_x = read.config["first_move_sigma_x"]
        self.first_sigma_y = read.config["first_move_sigma_y"]
        self.first_sigma_angle = read.config["first_move_sigma_angle"]
        self.resample_sigma_x = read.config["resample_sigma_x"]
        self.resample_sigma_y = read.config["resample_sigma_y"]
        self.resample_sigma_angle = read.config["resample_sigma_angle"]

        laser_sub = rospy.Subscriber(
            "/base_scan",
            LaserScan,
            self.sensorUpdate
        )
        self.likely_pub = rospy.Publisher(
            "/likelihood_field",
            OccupancyGrid,
            queue_size = 10,
            latch = True
        )
        self.pose_pub = rospy.Publisher(
            "/particlecloud",
            PoseArray,
            queue_size = 10
        )
        map_sub = rospy.Subscriber(
            "/map",
            OccupancyGrid,
            self.mapCallBack
        )
        rospy.sleep(1)
        self.width = self.map.width
        self.height = self.map.height

        # particle init
        for i in range(0,self.config["num_particles"]):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            theta = random.uniform(0, 2*m.pi)
            obstacle = self.map.get_cell(x,y)
            if(obstacle < 0.5):
                pose = get_pose(x,y,theta)
                self.pose_array.poses.append(pose)
            else:
                i -= 1
        rospy.sleep(1)
        self.pose_pub.publish(self.pose_array)
        
        self.moveRobot(self.moves)
        
        rospy.spin()

    def mapCallBack(self, message):
        # 2
        self.map = Map(message)
        self.width = self.map.width
        self.height = self.map.height
        sig = self.config['laser_sigma_hit']
        coordinates = []
        queryPts = []
        rospy.sleep(1)
        # 3
        for col in range(0, self.width):
            for row in range(0, self.height):
                x, y = self.map.cell_position(row, col)
                cellVal = self.map.get_cell(x,y)
                if (cellVal > 0.5):
                    coordinates.append([x,y])
        self.kdt = KDTree(coordinates)

        for col in range(0, self.width):
            for row in range(0, self.height):
                dist, ind = self.kdt.query([col,row], k=1)
                gauss = (1/(sig*m.sqrt(2*m.pi)))*((m.e)**-(((dist)**2)/(2*(sig**2))))
                self.map.set_cell(col, row, gauss)
        self.likely_pub.publish(self.map.to_message())

    def sensorUpdate(self, laser):
        self.laserMsg = laser

    def moveRobot(self, motion_cmd):
        for i in range(0, len(self.moves)): #4
            motion = self.moves[i]
            a = motion[0]
            d = motion[1]
            n = motion[2]
            move_function(a,0)            
            for t in range(0,n):
                move_function(0,d)
                updateParticles(a,d)

    def updateParticles(a,d):
        new_pose_array = []
        for j in range(0, self.config["num_particles"]):
            particle = self.pose_array.poses[j]
            x = particle[0]
            y = particle[1]
            theta = particle[2]
            if (self.isFirstMove):
                x = x + random.gauss(0,first_move_sigma_x)
                y = y + random.gauss(0,first_move_sigma_y)
                theta = theta + random.gauss(0,first_move_sigma_theta)
                self.isFirstMove = False
            x = x + d*math.cos(a)
            y = y + d*math.sin(a)
            theta = theta + a
            pose = get_pose(x, y, theta)
            new_pose_array.append(pose)
            
        self.pose_array.poses = new_pose_array
        #Reweight

    def motionUpdate(self, motion_cmd, particle):
        a = motion_cmd[0]
        d = motion_cmd[1]
        n = motion_cmd[2]
        
if __name__ == '__main__':
    Robot()
