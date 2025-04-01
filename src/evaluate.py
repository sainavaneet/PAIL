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

# Import your utility functions and robot state controller.
from utils import *
from controller.robot_state import *

# ROS services
from gazebo_msgs.srv import SetModelState, SetModelStateRequest 
from std_srvs.srv import Empty

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Initialize the CV bridge
bridge = CvBridge()

def set_box_position(x, y, z):
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    state_msg = SetModelStateRequest()
    state_msg.model_state.model_name = 'stone'
    state_msg.model_state.pose.position.x = x
    state_msg.model_state.pose.position.y = y
    state_msg.model_state.pose.position.z = z
    set_state(state_msg)

def generate_coordinate():
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

def image_callback(msg):
    try:
        # Extract the height and width from the message.
        height = msg.height
        width = msg.width
        
        # Create a NumPy array from the raw data.
        # We assume the encoding is "bgr8", so each pixel has 3 channels.
        np_img = np.frombuffer(msg.data, dtype=np.uint8)
        cv_image = np_img.reshape((height, width, 3))

        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Display the image
        cv2.imshow('Camera View', rgb_image)  # Display the RGB image
        cv2.waitKey(1)  # Refreshes the display window

    except Exception as e:
        rospy.logerr("Error converting ROS image message to OpenCV image: %s", e)
        return None
    return rgb_image



def capture_image(camera_name):
    camera_topics = {
        'top': '/fr3/camera/image_raw',
        'front': '/fr3/camera2/image_raw'
    }
    topic = camera_topics.get(camera_name, '/fr3/camera/color/image_raw')
    msg = rospy.wait_for_message(topic, Image, timeout=10)
    return image_callback(msg)

@hydra.main(config_path="/home/navaneet/vqvae_transformer/config", config_name="config")

def main(cfg: DictConfig):

    rospy.loginfo("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # Extract configuration sections
    train_cfg = cfg.train
    task_cfg = cfg.task
    policy_cfg = cfg.policy

    # Convert checkpoint directory to an absolute path (Hydra changes working directory)
    checkpoint_dir = to_absolute_path(train_cfg.checkpoint_dir)
    
    # Set the box position (ensure BOX_Z is defined appropriately)
    BOX_Z = 0.1  
    # set_box_position(0.45, -0.22, BOX_Z)
    rospy.sleep(1)

    # Initialize the robot controller and set to its initial pose
    franka = RobotController()
    franka.initial_pose()

    # Load the policy checkpoint
    ckpt_path = os.path.join(checkpoint_dir, train_cfg.eval_ckpt_name)
    device = torch.device('cuda')
    policy = make_policy(cfg.policy.policy_class, cfg)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()
    rospy.loginfo(f'Policy loaded from {ckpt_path}')

    # Load dataset statistics (for normalization/post-processing)
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    pre_process = lambda pos: (pos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda act: act * stats['action_std'] + stats['action_mean']

    # Determine query frequency and (if using temporal aggregation) number of queries
    query_frequency = policy_cfg.num_queries
    if policy_cfg.temporal_agg:
        query_frequency = 20
        num_queries = policy_cfg.num_queries
    else:
        num_queries = None  # Not used if temporal aggregation is disabled

    # Get the current robot state and gripper status
    position = franka.angles()
    gripper_width = franka.gripper_state()
    gripper_width = franka.gripper_state()
    if isinstance(gripper_width, tuple):
        gripper_width = gripper_width[0]  # Extract the first element

    gripper_width = 0 if gripper_width > 0.04 else 1

    # Prepare observation: current joint positions and camera images.
    pos = np.append(position, gripper_width)
    obs = {
        'qpos': pos,
        'images': {cn: capture_image(cn) for cn in task_cfg.camera_names}
    }

    n_rollouts = 1
    action_list = []

    # Allocate buffers for temporal aggregation if needed.
    if policy_cfg.temporal_agg:
        all_time_actions = torch.zeros(
            [task_cfg.episode_len, task_cfg.episode_len + num_queries, task_cfg.state_dim]
        ).to(device)
    qpos_history = torch.zeros((1, task_cfg.episode_len, task_cfg.state_dim)).to(device)

    # Begin rollout (evaluation loop)
    for i in tqdm(range(n_rollouts), desc=f"rollout: {n_rollouts}"):
        with torch.inference_mode():
            for t in range(task_cfg.episode_len):
                # Process the current joint positions.
                if isinstance(obs['qpos'], np.ndarray):
                    qpos_numpy = obs['qpos']
                elif isinstance(obs['qpos'], torch.Tensor):
                    qpos_numpy = obs['qpos'].cpu().numpy()
                else:
                    rospy.logerr("Unhandled data type for qpos: %s", type(obs['qpos']))
                    continue

                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos

                # Get the current image input (this function is assumed to be in your utils)
                curr_image = get_image(obs['images'], task_cfg.camera_names, device)

                
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)

                
                if policy_cfg.temporal_agg:
                    all_time_actions[t:t+1, t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[t, t:t+num_queries]
                    # Filter out zero actions if present.
                    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                # Post-process and store the action.
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action_list.append(action.tolist())

    # Save the actions to a CSV file.
    df = pd.DataFrame(action_list, columns=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"])
    df.to_csv('/home/navaneet/vqvae_transformer/actions_qil_close_lid.csv', index=False)
    rospy.loginfo("Actions have been saved to file.")

if __name__ == "__main__":
    main()
