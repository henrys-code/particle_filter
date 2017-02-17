#!/usr/bin/env python

import rospy
import math as m
import numpy as np
import random
from std_msgs.msg import Bool
from map_utils import Map
from copy import deepcopy
from read_config import read_config
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray, PointStamped, Quaternion, Point, Twist
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
        random.seed(self.config["seed"])
        self.move_list = self.config["move_list"]
        #Particle update values to use
        self.first_sigma_x = self.config["first_move_sigma_x"]
        self.first_sigma_y = self.config["first_move_sigma_y"]
        self.first_sigma_angle = self.config["first_move_sigma_angle"]
        self.resample_sigma_x = self.config["resample_sigma_x"]
        self.resample_sigma_y = self.config["resample_sigma_y"]
        self.resample_sigma_angle = self.config["resample_sigma_angle"]
        self.isFirstMove = True

        self.particleList = []
        self.posePtr = []
        self.tally = 0
        

        laser_sub = rospy.Subscriber(
            "/base_scan",
            LaserScan,
            self.senseCallBack
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
        self.result_pub = rospy.Publisher(
            "/result_update",
            Bool,
            queue_size = 10
        )
        self.sim_pub = rospy.Publisher(
            "/sim_complete",
            Bool,
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
        i = 0
        self.numParticles = self.config["num_particles"]
        while(i < self.numParticles):
            i += 1
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            theta = random.uniform(0, 2*m.pi)
            obstacle = self.map.get_cell(x,y)
            
            if(obstacle < 0.5):
                pose = get_pose(x,y,theta)
                self.pose_array.poses.append(pose)
                p = Particle(x,y,theta, 1)
                self.particleList.append(p)
            else:
                i -= 1
        self.pose_pub.publish(self.pose_array)
        rospy.wait_for_message("/likelihood_field", OccupancyGrid)
        self.moveRobot()
        self.sim_pub.publish(True)
        rospy.is_shutdown()
        rospy.spin()

    def sigmoid(self, x):
        y = (1.0 / (1.0 + m.exp(-x)))
        return y

    def mapCallBack(self, message):
        self.map = Map(message)
        self.whocares = Map(message)
        self.width = self.map.width
        self.height = self.map.height
        self.sig = self.config['laser_sigma_hit']
        coordinates = []
        rospy.sleep(1)

        for col in range(0, self.width):
            for row in range(0, self.height):
                x, y = self.map.cell_position(row, col)
                self.cellVal = self.map.get_cell(x,y)
                if (self.cellVal > 0.5):
                    coordinates.append([x,y])
        self.kdt = KDTree(coordinates)

        for col in range(0, self.width):
            for row in range(0, self.height):
                dist, ind = self.kdt.query([col,row], k=1)
                gauss = (1/(self.sig*m.sqrt(2*m.pi)))*((m.e)**-(((dist)**2)/(2*(self.sig**2))))
                self.map.set_cell(col, row, gauss)
        self.likely_pub.publish(self.map.to_message())

    def senseCallBack(self, laser):
        self.lsr = laser

    def moveRobot(self):
        for t in range(0, len(self.move_list)):
            move = self.move_list[t]
            #move robot
            a = move[0]
            d = move[1]
            n = move[2]
            move_function(a,0)
            self.first = True
            for s in range(0, n):
                move_function(0, d)
                self.motion_update(move)
                self.first = False
                self.isFirstMove = False
                self.pose_array.poses = self.posePtr
                self.pose_pub.publish(self.pose_array)
                self.posePtr = []
                self.sensor_update()
                self.resample()
            self.result_pub.publish(True)

    #For one particle
    def motion_update(self, mv):
        for i in range(0, len(self.particleList)):
            part_in = self.particleList[i]
            x = part_in.getX()
            y = part_in.getY()
            theta = part_in.getTheta()
            weight = part_in.getWeight()
            if (self.isFirstMove):
                x = x + random.gauss(0, self.first_sigma_x)
                y = y + random.gauss(0, self.first_sigma_y)
                theta = theta + random.gauss(0, self.first_sigma_angle)
            if (self.first):
                theta = theta + m.radians(mv[0])
            x = x + mv[1] * m.cos(theta)
            y = y + mv[1] * m.sin(theta)
            new_pose = get_pose(x,y,theta)
            self.posePtr.append(new_pose)
            part_in.setX(x)
            part_in.setY(y)
            part_in.setTheta(theta)    
            if (m.isnan(self.whocares.get_cell(x,y)) or self.whocares.get_cell(x,y) == 1.0):
                part_in.setWeight(0.0)

    def sensor_update(self):
        z_rand = self.config["laser_z_rand"]
        z_hit = self.config["laser_z_hit"]
        normalizer = 0
        #Laser Scan
        for p in range(0, len(self.particleList)):
            particle = self.particleList[p]
            if(self.map.get_cell(particle.getX(),particle.getY()) == 1):
                particle.setWeight(0.0)
            ptot = 0
            noNAN = 0
            for l in range(0,len(self.lsr.ranges)):
                scan = self.lsr.ranges[l]
                if(scan != self.lsr.range_min and scan != self.lsr.range_max):
                    angle = self.lsr.angle_min + (self.lsr.angle_increment * l) + particle.getTheta()
                    x_pos = scan * m.cos(angle) + particle.getX()
                    y_pos = scan * m.sin(angle) + particle.getY()
                    lp = self.map.get_cell(x_pos,y_pos)
                    if (m.isnan(lp)):
                        noNAN += 1
                    else: 
                        if (noNAN > (len(self.lsr.ranges)*.75)):
                            #particle.setWeight(0.0)
                            x = random.randint(0, self.width)
                            y = random.randint(0, self.height)
                            theta = random.uniform(0, 2*m.pi)
                        pz = (z_hit * lp) + z_rand
                        ptot += (pz**3)
                    
            old_weight = particle.getWeight() 
            new_weight = old_weight * self.sigmoid(ptot)
            normalizer += new_weight
            new_particle = Particle(particle.getX(),particle.getY(),particle.getTheta(),new_weight)
            self.particleList[p] = new_particle
        
        for q in range(0, len(self.particleList)):
            particle = self.particleList[q]
            weight = particle.getWeight() / normalizer
            particle.setWeight(weight)

    def resample(self):
        weights = []
        good_weights = []
        good_particles = []
        for w in range(0, len(self.particleList)):
            particle = self.particleList[w]
            value = self.whocares.get_cell(particle.getX(),particle.getY())
            if (not (value == 1.0 or m.isnan(value))):
                weight100 = particle.getWeight() * 100
                it = 0
                while (it < weight100):
                    weights.append(particle)
                    it += 1
                good_particles.append(self.particleList[w])
                good_weights.append(self.particleList[w].getWeight())
        
        # new_particles = np.random.choice(self.particleList, self.numParticles, True, weights)
        for p in range(0,len(self.particleList)):
            particle = self.particleList[p]
            value = self.whocares.get_cell(particle.getX(),particle.getY())
            if (value == 1.0 or m.isnan(value)):
                choice = int(random.uniform(0,len(weights)))
                particle = weights[choice]
                #dingledangle = np.random.choice(good_particles, 1, True, good_weights)
                #particle = dingledangle[0]
                
            x = particle.getX() + random.gauss(0,self.resample_sigma_x)
            y = particle.getY() + random.gauss(0,self.resample_sigma_y)
            theta = particle.getTheta() + random.gauss(0,self.resample_sigma_angle)
            new_particle = Particle(x,y,theta,particle.getWeight())
            self.particleList[p] = new_particle
         

class Particle():
    def __init__(self, x_in, y_in, theta_in, weight_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.weight = weight_in

    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getTheta(self):
        return self.theta
    def getWeight(self):
        return self.weight

    def setX(self, x_in):
        self.x = x_in
    def setY(self, y_in):
        self.y = y_in
    def setTheta(self, theta_in):
        self.theta = theta_in
    def setWeight(self, weight_in):
        self.weight = weight_in
 
if __name__ == '__main__':
    Robot()
