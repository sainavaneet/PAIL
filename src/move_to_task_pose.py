#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import sys
sys.path.append('/home/navaneet/vqvae_transformer/controller')

from robot_state import *
import time

robot = RobotController()



robot.move_to_joint_position([0.05415077394146999, -0.06999536879369829, 0.025047756366318442, -2.0467223351719332, 0.006527885864619271, 1.9560050275723249, 2.4006788392637186])
