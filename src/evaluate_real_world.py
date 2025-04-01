import os
import random
import subprocess
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

from utils import *
from controller.robot_state import *

# ROS services
from gazebo_msgs.srv import SetModelState, SetModelStateRequest 
from std_srvs.srv import Empty

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import sys

sys.path.append('/home/navaneet/vqvae_transformer/controller')


class RobotPolicyRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.task_cfg = cfg.task
        self.policy_cfg = cfg.policy
        rospy.loginfo("Initializing Robot Policy Runner...")
        self.franka = RobotController()
        self.bridge = CvBridge()
        self.gripper_grasp_position = False
        self.release_all_cameras()
        self.franka.move_to_joint_position(self.cfg.real_world.demonstration.task_pose)
        self.action_counter = 0  # Add a counter to track actions
        
        self.cameras = {
            'top': cv2.VideoCapture(6), 
            'front': cv2.VideoCapture(0)
        }
        for cam in self.cameras.values():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cam.isOpened():
                rospy.logerr("Error: Could not open camera.")
        # Set checkpoint directory and load policy
        checkpoint_dir = to_absolute_path(self.train_cfg.checkpoint_dir)
        self.ckpt_path = os.path.join(checkpoint_dir, self.train_cfg.eval_ckpt_name)
        self.device = torch.device('cuda')
        self.policy = make_policy(self.policy_cfg.policy_class, cfg)
        self.policy.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()
        rospy.loginfo(f'Policy loaded from {self.ckpt_path}')

        # Load stats and create processing functions
        stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        self.pre_process = lambda pos: (pos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda act: act * self.stats['action_std'] + self.stats['action_mean']


    def release_all_cameras(self):
        """Release all previously enabled cameras."""
        for i in [0 , 6]:  
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                cam.release()

    def set_box_position(self, x, y, z):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'stone'
        state_msg.model_state.pose.position.x = x
        state_msg.model_state.pose.position.y = y
        state_msg.model_state.pose.position.z = z
        set_state(state_msg)

    def generate_coordinate(self):
        box_length = 0.18
        box_width = 0.11
        box_x_center = 0.45
        box_y_center = -0.21
        cube_x = 0.025
        cube_y = 0.032
        min_x = box_x_center - box_length / 2 + cube_x / 2
        max_x = box_x_center + box_length / 2 - cube_x / 2
        min_y = box_y_center - box_width / 2 + cube_y / 2
        max_y = box_y_center + box_width / 2 - cube_y / 2

        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        return x, y

    def capture_image(self, camera_name):
        cam = self.cameras.get(camera_name)
        if cam is None or not cam.isOpened():
            rospy.logerr(f"Invalid camera name or camera not opened: {camera_name}")
            return None
        
        ret, frame = cam.read()
        if not ret:
            rospy.logerr(f"Failed to capture image from camera {camera_name}")
            return None
        
        # cv2.imshow(f"Camera: {camera_name}", frame)
        # cv2.waitKey(1)  # Wait for 1 ms to allow the window to refresh

        # rospy.loginfo(f"Successfully captured image from camera {camera_name}")
        return frame

    def action_publisher(self, action):

        self.action_counter += 1

        # Skip every other action
        if self.action_counter % 2 == 0:
            # rospy.loginfo("Skipping this action.")
            return

        joint_positions = action[:7].tolist()
        gripper_position = action[7]

        self.franka.move_to_joint_position(joint_positions)

        if not self.gripper_grasp_position:
            if gripper_position <= 0.058:
                self.franka.exec_gripper_cmd(self.cfg.real_world.demonstration.gripper.close, 1)
                self.gripper_grasp_position = True

    def run(self):
        rospy.loginfo("Starting main run loop.")

        captured_images = []
        for i in range(10):
            top_image = self.capture_image('top')
            front_image = self.capture_image('front')
            if top_image is not None and front_image is not None:
                captured_images.append((top_image, front_image))
            rospy.loginfo(f"Captured image pair {i + 1}/10")

        # Delete the captured images
        del captured_images
        rospy.loginfo("Deleted all captured images.")

        rospy.sleep(3)
        self.franka.move_to_joint_position(self.cfg.real_world.demonstration.task_pose)
        rospy.sleep(3)

        self.franka.exec_gripper_cmd(self.cfg.real_world.demonstration.gripper.open, 1)
        rospy.sleep(1)

        query_frequency = 20
        num_queries = self.policy_cfg.num_queries if self.policy_cfg.temporal_agg else None

        n_rollouts = 1
        action_list = []

        if self.policy_cfg.temporal_agg:
            all_time_actions = torch.zeros([
                self.task_cfg.episode_len, self.task_cfg.episode_len + num_queries, self.task_cfg.state_dim
            ]).to(self.device)
        qpos_history = torch.zeros((1, self.task_cfg.episode_len, self.task_cfg.state_dim)).to(self.device)

        for i in tqdm(range(n_rollouts), desc=f"rollout: {n_rollouts}"):
            with torch.inference_mode():
                for t in range(self.task_cfg.episode_len):
                    position = self.franka.angles()
                    gripper_width = self.franka.gripper_width()
                    pos = np.append(position, gripper_width)
                    obs = {
                        'qpos': pos,
                        'images': {
                            'top': self.capture_image('top'),
                            'front': self.capture_image('front')
                        }
                    }
                    qpos_numpy = obs['qpos'] if isinstance(obs['qpos'], np.ndarray) else obs['qpos'].cpu().numpy()
                    qpos = self.pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().to(self.device).unsqueeze(0)
                    curr_image = get_image(obs['images'], self.task_cfg.camera_names, self.device)
                    
                    if self.policy_cfg.policy_class == "QIL":
                        if t % query_frequency == 0:
                            all_actions = self.policy(qpos, curr_image)
                        if self.policy_cfg.temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights /= exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    else:
                        raise NotImplementedError

                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = self.post_process(raw_action)
                    print(f"Action {t} : {action}")

                    self.action_publisher(action)
                    action_list.append(action.tolist())
        self.franka.exec_gripper_cmd(self.cfg.real_world.demonstration.gripper.open, 1)
        df = pd.DataFrame(action_list, columns=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"])
        output_path = os.path.join(to_absolute_path(self.train_cfg.checkpoint_dir), 'actions_test_ring_insertion.csv')
        df.to_csv(output_path, index=False)
        rospy.loginfo(f"Actions have been saved to file: {output_path}")

    def __del__(self):
        for cam in self.cameras.values():
            cam.release()
        cv2.destroyAllWindows()
        
@hydra.main(config_path="/home/navaneet/vqvae_transformer/config", config_name="config")
def main(cfg: DictConfig):
    # rospy.loginfo("Configuration:\n" + OmegaConf.to_yaml(cfg))
    runner = RobotPolicyRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
