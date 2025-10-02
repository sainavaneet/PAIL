#!/usr/bin/env python3
"""
Script to reorganize libero_90 video files into a flat structure.
Each task folder contains episode files that will be copied to a single folder
with only episode_0 files renamed to match task names.
"""

import os
import shutil
from pathlib import Path

def reorganize_libero_videos_flat():
    # Source directory
    source_dir = Path("/home/navaneet/DAIL/static/videos/libero/bak/libero_goal/videos")
    
    # Destination directory
    dest_dir = Path("/home/navaneet/DAIL/static/videos/libero/libero_goal")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all task folders
    task_folders = [f for f in source_dir.iterdir() if f.is_dir()]
    
    print(f"Found {len(task_folders)} task folders")
    
    for task_folder in task_folders:
        task_name = task_folder.name
        print(f"Processing task: {task_name}")
        
        # Look for episode_0.mp4 in the task folder
        episode_0_file = task_folder / "episode_0.mp4"
        
        if not episode_0_file.exists():
            print(f"  No episode_0.mp4 found in {task_name}")
            continue
            
        # Create new filename: task_name.mp4
        new_filename = f"{task_name}.mp4"
        dest_file = dest_dir / new_filename
        
        # Copy the file
        shutil.copy2(episode_0_file, dest_file)
        print(f"    Copied: episode_0.mp4 -> {new_filename}")
    
    print(f"\nReorganization complete! Files saved to: {dest_dir}")
    print(f"Total tasks processed: {len(task_folders)}")

if __name__ == "__main__":
    reorganize_libero_videos_flat()
