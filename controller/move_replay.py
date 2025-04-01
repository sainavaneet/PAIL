import cv2
import h5py
from robot_state import RobotController

franka = RobotController()
franka.initial_pose()

def play_images_from_hdf5_opencv(file_path, group_name, dataset_name, joint_positions_dataset, frame_rate=30, skip_interval=10):
    interval = int(1000 / frame_rate)
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        images_dataset = file[group_name][dataset_name]
        joint_positions = file[group_name][joint_positions_dataset]

        # cv2.namedWindow('HDF5 Image Sequence', cv2.WINDOW_NORMAL)
        
        for i in range(images_dataset.shape[0]):
            if i % skip_interval != 0:  
                continue

            image = images_dataset[i, ...]
            if image.dtype != 'uint8':
                image = image.astype('uint8')

            # cv2.imshow('HDF5 Image Sequence', image)
    
            joint_pos = joint_positions[i]
            
            print(i)
            print(joint_pos)
            
            franka.move_to_joint_position(joint_pos[:7])
            franka.exec_gripper_cmd(joint_pos[7])

    
            key = cv2.waitKey(interval)
            if key == 27:  # Escape key to exit
                break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    hdf5_path = f'/home/navaneet/ACTfranka/output.hdf5'
    group_name = '/data/demo_6/obs'
    images_dataset_name = 'agentview_rgb'
    joint_positions_dataset_name = 'joint_states'
    frame_rate = 30
    skip_interval = 2  # Define the skip interval
    play_images_from_hdf5_opencv(hdf5_path, group_name, images_dataset_name, joint_positions_dataset_name, frame_rate, skip_interval)
