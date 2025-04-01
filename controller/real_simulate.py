import pandas as pd
import numpy as np
import random
import rospy
from robot_state import RobotController
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from rich.progress import Progress

import progressbar

class Simulator:
    def __init__(self, file_path='/home/navaneet/vqvae_transformer/actions_qil.csv'):
        
        self.franka = RobotController()
        self.data = pd.read_csv(file_path, header=None, skiprows=1)
        self.gripper_grasp_position = False
        rospy.sleep(1)

        self.franka.move_to_joint_position([0.05415077394146999, -0.06999536879369829, 0.025047756366318442, -2.0467223351719332, 0.006527885864619271, 1.9560050275723249, 2.4006788392637186]
)

        self.franka.exec_gripper_cmd(0.06, 1)



    def simulate(self):
        with Progress() as progress:
            task = progress.add_task("[green]Simulating...", total=len(self.data))
            
            for index, row in self.data.iterrows():
                print(index)
                
                progress.update(task, advance=1)
            
                if index % 1 == 0:
                    
                    joint_positions = row[:7].tolist()
                    gripper_position = row[7]

                    self.franka.move_to_joint_position(joint_positions)

                    if not self.gripper_grasp_position:
                        if gripper_position < 0.044:
                            self.franka.exec_gripper_cmd(0.035, 1)
                            print("Grasping")
                            self.gripper_grasp_position = True
                        else:
                            self.franka.exec_gripper_cmd(0.06, 1)
            self.franka.exec_gripper_cmd(0.06, 1)
                    
            


if __name__ == "__main__":
    try:
    
        simulator = Simulator(file_path='/home/navaneet/vqvae_transformer/dataset/assets/real_world/ring_insertion/test_checkpoints/actions_test_ring_insertion.csv')  
        simulator.simulate()
    
    except rospy.ROSInterruptException:
        pass