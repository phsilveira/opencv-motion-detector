#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/run_tests.py
import os
import sys
import time
import glob
import signal
import shutil
import logging
import subprocess
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestRunner")

def clean_folder(folder_path):
    """Clean all image files from the specified folder"""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        logger.info(f"Cleaning folder: {folder_path}")
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")
        logger.info(f"Folder {folder_path} cleaned")
    else:
        logger.warning(f"Folder {folder_path} does not exist")

def find_video_files(path_pattern):
    """Find all video files matching the given pattern"""
    video_files = glob.glob(path_pattern)
    if not video_files:
        logger.error(f"No video files found matching pattern: {path_pattern}")
        sys.exit(1)
    return video_files

def run_motion_watcher(min_area_percent=1, enable_analysis=True, persistence=10):
    """Start the motion watcher process"""
    cmd = [
        sys.executable, 
        "motion_watcher.py", 
        "-m", str(min_area_percent),
        "-p", str(persistence),
        "-n", "4",
    ]
    
    if enable_analysis:
        cmd.append("--enable-analysis")
    
    logger.info(f"Starting motion_watcher with command: {' '.join(cmd)}")
    
    # Start motion_watcher process
    process = subprocess.Popen(cmd)
    
    # Wait a bit to ensure motion_watcher is fully started
    time.sleep(2)
    
    return process

def run_camera_capture(video_file, fps=1):
    """Run camera_capture on a specific video file"""
    cmd = [
        sys.executable,
        "camera_capture.py",
        "--fps", str(fps),
        "-v", video_file,
        "-s", "2.0",
    ]
    
    logger.info(f"Processing video: {video_file}")
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run camera_capture process and wait for it to complete
    try:
        process = subprocess.Popen(cmd)
        process.wait()  # Wait for the process to complete
        
        if process.returncode != 0:
            logger.error(f"camera_capture process failed with return code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running camera_capture: {e}")
        return False

def run_tests(video_pattern, min_area_percent=1, enable_analysis=True, fps=1, persistence=10):
    """Run the full test process"""
    camera_frames_dir = "camera_frames"
    
    # Ensure camera_frames directory exists
    if not os.path.exists(camera_frames_dir):
        os.makedirs(camera_frames_dir)
    
    # Clean any existing frames
    clean_folder(camera_frames_dir)
    
    # Find video files
    video_files = find_video_files(video_pattern)
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Start the motion_watcher process
    motion_watcher_process = run_motion_watcher(
        min_area_percent=min_area_percent,
        enable_analysis=enable_analysis,
        persistence=persistence
    )
    
    try:
        # Process each video file
        for video_file in video_files:
            logger.info(f"Testing with video: {video_file}")
            
            # Run camera_capture on this video file
            success = run_camera_capture(video_file, fps=fps)
            
            if not success:
                logger.error(f"Failed to process video: {video_file}")
                continue
            
            # Allow time for motion_watcher to process remaining frames
            logger.info("Waiting for motion_watcher to process any remaining frames...")
            time.sleep(5)
            
            # Clean frames after each video to avoid confusion between tests
            clean_folder(camera_frames_dir)
    
    finally:
        # Terminate motion_watcher process
        if motion_watcher_process:
            logger.info("Stopping motion_watcher process...")
            motion_watcher_process.send_signal(signal.SIGINT)
            try:
                motion_watcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("motion_watcher did not terminate gracefully, forcing termination")
                motion_watcher_process.kill()
        
        # Final cleanup
        clean_folder(camera_frames_dir)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run motion detection tests on video files')
    
    parser.add_argument('-v', '--video-pattern', default='test_occurrences/video/*.mp4',
                        help='Pattern to match video files (default: test_occurrences/video/*.mp4)')
    
    parser.add_argument('-m', '--min-area', type=float, default=1.0,
                        help='Minimum area percentage for motion detection (default: 1.0)')
    
    parser.add_argument('--no-analysis', action='store_true',
                        help='Disable event analysis')
    
    parser.add_argument('-p', '--persistence', type=int, default=10,
                        help='Motion persistence value (default: 10)')
    
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to capture (default: 1.0)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Log the test start
    logger.info("=== Motion Detection Test Runner ===")
    logger.info(f"Starting tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Video pattern: {args.video_pattern}")
    logger.info(f"Min area percent: {args.min_area}")
    logger.info(f"Event analysis: {'Disabled' if args.no_analysis else 'Enabled'}")
    logger.info(f"Persistence: {args.persistence}")
    logger.info(f"Capture FPS: {args.fps}")
    
    # Run the tests
    run_tests(
        video_pattern=args.video_pattern,
        min_area_percent=args.min_area,
        enable_analysis=not args.no_analysis,
        fps=args.fps,
        persistence=args.persistence
    )
    
    logger.info("=== Test run completed ===")