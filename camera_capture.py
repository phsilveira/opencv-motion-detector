#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera Capture Script

This script continuously captures frames from a camera source, RTSP stream, or video file
and saves them to a specified directory at a rate of 1 frame per second.

Usage:
    python camera_capture.py [-c CAMERA_INDEX | -r RTSP_URL | -v VIDEO_FILE] [-o OUTPUT_FOLDER] [-s PLAYBACK_SPEED] [-f]
"""

import argparse
import cv2
import os
import time
from datetime import datetime
import sys


class CameraCapture:
    def __init__(self, source="rtsp://admin:rad001877@rc-uchida.dyndns.org:554/Streaming/Channels/102/", 
                 output_folder="camera_frames", playback_speed=1.0, fast_extract=False, save_fps=1):
        """Initialize the camera capture with specified parameters.
        
        Args:
            source: Camera index (int), RTSP URL (str), or video file path (str)
            output_folder: Directory to save captured frames
            playback_speed: Speed multiplier for video file playback (doesn't affect save rate)
            fast_extract: When True, skips display and processes video files at maximum speed (video files only)
            save_fps: Frames per second to save (default: 1)
        """
        self.source = source
        self.output_folder = output_folder
        self.cap = None
        self.frame_rate = save_fps  # Number of frames to save per second
        self.playback_speed = playback_speed
        self.fast_extract = fast_extract
        self.last_save_time = 0
        self.frame_counter = 0  # Sequential counter for all saved frames
        
        # Determine source type
        self.is_rtsp = isinstance(source, str) and source.lower().startswith('rtsp://')
        self.is_video_file = (isinstance(source, str) and 
                             not source.lower().startswith('rtsp://') and 
                             os.path.exists(source))
        self.is_camera = not (self.is_rtsp or self.is_video_file)
        
        # Extract source name for video files to use in output filenames
        self.source_name = "camera"
        if self.is_video_file:
            # Get just the filename without path and extension
            self.source_name = os.path.splitext(os.path.basename(source))[0]
        elif self.is_rtsp:
            # For RTSP, use a simplified name
            self.source_name = "rtsp_stream"
        
        # Print info about source
        if self.is_rtsp:
            print(f"Using RTSP stream: {source}")
        elif self.is_video_file:
            print(f"Using video file: {source}")
            if self.fast_extract:
                print("Fast extraction mode: Enabled (display disabled)")
        else:
            print(f"Using camera index: {source}")

        # Create output directory if it doesn't exist
        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"Created output directory: {self.output_folder}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)

    def setup_camera(self):
        """Set up the video source for capturing."""
        try:
            # First try to release any existing camera
            if self.cap is not None:
                self.cap.release()
            
            # For camera indices, flush the stream first
            if self.is_camera:
                temp_cap = cv2.VideoCapture(self.source)
                temp_cap.release()
            
            # Then start the capture
            self.cap = cv2.VideoCapture(self.source)
            
            # Set additional parameters for RTSP streams to reduce latency
            if self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size to reduce latency
                
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {self.source}")
            
            # Get video properties
            if self.is_video_file:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.duration = self.total_frames / self.fps if self.fps > 0 else 0
                print(f"Video file loaded: {self.source}")
                print(f"Total frames: {self.total_frames}, FPS: {self.fps:.2f}, Duration: {self.duration:.2f}s")
                print(f"Playback speed: {self.playback_speed}x")
            elif self.is_rtsp:
                print(f"RTSP stream initialized: {self.source}")
            else:
                print(f"Camera initialized with index {self.source}")
            
        except Exception as e:
            print(f"Error setting up video source: {e}")
            sys.exit(1)

    def save_frame(self, frame, frame_number=None):
        """Save a frame to the output directory with timestamp in the filename.
        
        Args:
            frame: The OpenCV frame to save
            frame_number: Optional frame number to include in the filename (used in fast extraction mode)
        """
        try:
            # Generate a timestamp string for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Increment frame counter for each saved frame
            self.frame_counter += 1
            
            # For fast extraction mode, include frame number to prevent overwriting
            if frame_number is not None:
                filename = os.path.join(self.output_folder, f"{self.source_name}_frame_{frame_number:06d}_{self.frame_counter:06d}.jpg")
            else:
                filename = os.path.join(self.output_folder, f"{self.source_name}_frame_{timestamp}_{self.frame_counter:06d}.jpg")
            
            # Save the frame
            cv2.imwrite(filename, frame)
            print(f"Frame saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None

    def run(self):
        """Main loop for camera capture."""
        try:
            self.setup_camera()
            print(f"Starting capture at {self.frame_rate} fps to folder: {self.output_folder}")
            if not self.fast_extract or not self.is_video_file:
                print("Press 'q' to quit")
            
            reconnect_attempts = 0
            max_reconnect_attempts = 5
            
            # For video files, calculate the frame interval for playback speed
            if self.is_video_file:
                frame_interval = 1.0 / (self.fps * self.playback_speed) if self.fps > 0 else 0.033
            else:
                frame_interval = 0.033  # ~30fps for cameras and RTSP
            
            frame_count = 0
            video_position = 0.0  # Position in seconds for video files
            frames_to_skip = 0
            
            # For fast extraction of video files
            if self.is_video_file and self.fast_extract:
                frames_per_second = int(self.fps)
                if frames_per_second <= 0:
                    frames_per_second = 30  # Default if can't determine FPS
                
                # Calculate how many source frames to skip to achieve desired save_fps
                frames_to_save = int(self.frame_rate)  # How many frames to save per second of video
                frame_interval = max(1, frames_per_second // frames_to_save)  # How many frames to skip between saves
                
                # Progress tracking for fast extraction
                last_progress_time = time.time()
                start_time = time.time()
                saved_frames = 0
                video_second = 0
                frame_within_second = 0
                
                print(f"Fast extraction mode: Processing video at {self.fps:.2f} fps, saving {self.frame_rate} frame(s) per second")
                
                while True:
                    # Read frame
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print("End of video file reached.")
                        break
                    
                    frame_count += 1
                    
                    # Calculate current second of video and position within that second
                    current_video_second = int(frame_count / frames_per_second)
                    
                    # Check if we've moved to a new second
                    if current_video_second > video_second:
                        video_second = current_video_second
                        frame_within_second = 0  # Reset counter for frames within this second
                    
                    # Determine if we should save this frame based on the desired save FPS
                    # Save frames at evenly spaced intervals to achieve the requested save_fps
                    should_save = False
                    
                    if frames_per_second <= frames_to_save:
                        # If video FPS is lower than save_fps, save every frame
                        should_save = True
                    else:
                        # Save if this frame is at a position that should be saved
                        frame_pos_in_second = frame_count % frames_per_second
                        for i in range(frames_to_save):
                            save_point = (i * frames_per_second) // frames_to_save
                            if frame_pos_in_second == save_point:
                                should_save = True
                                break
                    
                    if should_save:
                        # For multiple frames per second, add frame index within second
                        if self.frame_rate > 1:
                            frame_within_second += 1
                            frame_num = (video_second * self.frame_rate) + frame_within_second
                        else:
                            frame_num = video_second
                            
                        # Save with frame number to prevent overwriting
                        self.save_frame(frame, frame_num)
                        saved_frames += 1
                        
                        # Show progress every 5 seconds or for every 10% completion
                        current_time = time.time()
                        elapsed = current_time - start_time
                        progress = frame_count / self.total_frames if self.total_frames > 0 else 0
                        
                        if (current_time - last_progress_time >= 5.0) or (progress >= 0.1 and int(progress*10) > int((progress-0.01)*10)):
                            last_progress_time = current_time
                            eta = (elapsed / progress) - elapsed if progress > 0 else 0
                            print(f"Progress: {progress*100:.1f}% | Frames: {frame_count}/{self.total_frames} | " +
                                  f"Saved: {saved_frames} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s")
                
                print(f"Fast extraction complete. Saved {saved_frames} frames in {time.time() - start_time:.2f} seconds")
                return
            
            # Regular processing loop for live display
            while True:
                loop_start_time = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.is_video_file:
                        print("End of video file reached.")
                        break
                    else:
                        print("Error reading frame from source")
                        reconnect_attempts += 1
                        
                        if reconnect_attempts > max_reconnect_attempts:
                            print(f"Failed to reconnect after {max_reconnect_attempts} attempts")
                            break
                        
                        print(f"Attempting to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                        time.sleep(2)  # Wait before trying to reconnect
                        self.setup_camera()
                        continue
                
                # Reset reconnect counter on successful frame read
                reconnect_attempts = 0
                
                # Update video position for video files
                if self.is_video_file:
                    frame_count += 1
                    video_position = frame_count / self.fps
                
                # Get current time
                current_time = time.time()
                
                # Save frame based on the frame_rate setting
                if current_time - self.last_save_time >= 1.0 / self.frame_rate:
                    self.save_frame(frame)
                    self.last_save_time = current_time
                
                # Display the frame with timestamp and information overlay
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, current_datetime, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add source type to display
                if self.is_rtsp:
                    source_type = "RTSP Stream"
                elif self.is_video_file:
                    source_type = f"Video File ({video_position:.1f}s / {self.duration:.1f}s)"
                else:
                    source_type = f"Camera {self.source}"
                    
                cv2.putText(frame, f"Source: {source_type}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Saving to: {self.output_folder}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # If it's a video file, show playback speed
                if self.is_video_file:
                    cv2.putText(frame, f"Playback: {self.playback_speed}x", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow("Frame Capture (1 FPS Save)", frame)
                
                # Check for quit command and playback speed adjustments
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    if self.is_video_file:
                        self.playback_speed = min(10.0, self.playback_speed * 1.25)
                        print(f"Playback speed increased to {self.playback_speed:.2f}x")
                elif key == ord('-'):
                    if self.is_video_file:
                        self.playback_speed = max(0.25, self.playback_speed * 0.8)
                        print(f"Playback speed decreased to {self.playback_speed:.2f}x")
                
                # Calculate how long to wait to maintain proper playback speed
                # (only relevant for video files)
                if self.is_video_file:
                    elapsed = time.time() - loop_start_time
                    frame_interval = 1.0 / (self.fps * self.playback_speed) if self.fps > 0 else 0.033
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("Capture interrupted by user")
        except Exception as e:
            print(f"Error in camera capture: {e}")
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Camera capture stopped")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Camera frame capture at 1 FPS')
    
    # Create a mutually exclusive group for video source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('-c', '--camera', type=int,
                        help='Camera device index')
    source_group.add_argument('-r', '--rtsp', type=str,
                        default="rtsp://admin:rad001877@rc-uchida.dyndns.org:554/Streaming/Channels/102/",
                        help='RTSP URL (default: rtsp://admin:rad001877@rc-uchida.dyndns.org:554/Streaming/Channels/102/)')
    source_group.add_argument('-v', '--video', type=str,
                        help='Path to video file')
    
    parser.add_argument('-o', '--output-folder', default='camera_frames',
                        help='Folder to save captured frames to (default: camera_frames)')
    parser.add_argument('-s', '--speed', type=float, default=1.0,
                        help='Playback speed for video files (default: 1.0)')
    parser.add_argument('-f', '--fast', action='store_true',
                        help='Enable fast extraction mode for video files (no display, maximum speed)')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to save (default: 1.0)')
    
    return parser.parse_args()


def main():
    """Main entry point of the application."""
    args = parse_arguments()
    
    # Determine the source
    if args.camera is not None:
        source = args.camera
    elif args.video is not None:
        source = args.video
    else:
        source = args.rtsp
    
    capture = CameraCapture(
        source=source,
        output_folder=args.output_folder,
        playback_speed=args.speed,
        fast_extract=args.fast,
        save_fps=args.fps
    )
    
    capture.run()


if __name__ == "__main__":
    main()
