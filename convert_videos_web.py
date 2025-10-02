#!/usr/bin/env python3
"""
Script to convert libero_90_flat videos to web-compatible H.264 format.
Converts MPEG-4 videos to H.264 for better browser and VS Code compatibility.
"""

import os
import subprocess
from pathlib import Path

def convert_videos_to_web_compatible():
    # Source directory (flat structure)
    source_dir = Path("/home/navaneet/DAIL/static/videos/libero/libero_goaflat")
    
    # Destination directory
    dest_dir = Path("/home/navaneet/DAIL/static/videos/libero/libero_goal")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all MP4 files
    mp4_files = list(source_dir.glob("*.mp4"))
    
    print(f"Found {len(mp4_files)} MP4 files to convert")
    
    for i, video_file in enumerate(mp4_files, 1):
        print(f"\n[{i}/{len(mp4_files)}] Converting: {video_file.name}")
        
        # Create output filename
        output_file = dest_dir / video_file.name
        
        # FFmpeg command to convert to H.264 with web-optimized settings
        cmd = [
            'ffmpeg',
            '-i', str(video_file),           # Input file
            '-c:v', 'libx264',              # Video codec: H.264
            '-preset', 'medium',            # Encoding speed vs compression
            '-crf', '23',                   # Constant rate factor (quality)
            '-c:a', 'aac',                  # Audio codec: AAC
            '-movflags', '+faststart',      # Optimize for web streaming
            '-pix_fmt', 'yuv420p',         # Pixel format for compatibility
            '-y',                           # Overwrite output files
            str(output_file)                # Output file
        ]
        
        try:
            # Run the conversion
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"    ✓ Converted successfully")
            
            # Show file size comparison
            original_size = video_file.stat().st_size
            converted_size = output_file.stat().st_size
            size_change = ((converted_size - original_size) / original_size) * 100
            
            print(f"    Original: {original_size / 1024 / 1024:.1f} MB")
            print(f"    Converted: {converted_size / 1024 / 1024:.1f} MB ({size_change:+.1f}%)")
            
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Error converting {video_file.name}: {e}")
            print(f"    Error output: {e.stderr}")
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
    
    print(f"\nConversion complete! Web-compatible videos saved to: {dest_dir}")
    print(f"Total files processed: {len(mp4_files)}")

if __name__ == "__main__":
    convert_videos_to_web_compatible()
