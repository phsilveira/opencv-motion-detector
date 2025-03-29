#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenCV Motion Detector
@original author: methylDragon
@refactored by: GitHub Copilot

Description:
This script detects motion from a video source (webcam or video file).
When movement is detected and large enough, it reports motion in the frame.
"""

import argparse
import imutils
import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime


class MotionDetector:
    def __init__(self, config):
        """Initialize the motion detector with the given configuration."""
        # Configuration parameters
        self.frames_to_persist = config.get('frames_to_persist', 10)
        self.min_size_for_movement = config.get('min_size_for_movement', 2000)
        self.movement_detected_persistence = config.get('movement_detected_persistence', 100)
        self.threshold = config.get('threshold', 25)  # Default threshold value
        self.minimum_motion_frames = config.get('minimum_motion_frames', 1)  # Minimum consecutive frames with motion
        self.save_frames = config.get('save_frames', False)  # Whether to save frames when motion is detected
        self.output_folder = config.get('output_folder', 'motion_detection')  # Folder to save frames to
        self.motion_framerate = config.get('motion_framerate', 15)  # Framerate for motion capture
        
        # Initialize frame variables
        self.first_frame = None
        self.next_frame = None
        self.delay_counter = 0
        self.movement_persistent_counter = 0
        self.consecutive_motion_frames = 0  # Track consecutive frames with motion
        
        # Calculate frame interval in seconds based on framerate
        self.frame_interval = 1.0 / max(2, self.motion_framerate)  # Ensure minimum of 2 fps
        
        # Initialize timestamp for saving frames
        self.last_saved_time = 0
        
        # Create output directory if it doesn't exist and frame saving is enabled
        if self.save_frames:
            self._create_output_directory()
        
        # Initialize capture object
        self.cap = None
        self.setup_video_source(config.get('source', 0))
        
        # Set font for text overlay
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _create_output_directory(self):
        """Create the output directory for saving frames if it doesn't exist."""
        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"Created output directory: {self.output_folder}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            self.save_frames = False  # Disable frame saving if the directory cannot be created

    def _save_frame(self, frame):
        """Save a frame to the output directory with timestamp in the filename."""
        if not self.save_frames:
            return
        
        # Check if it's time to save another frame (based on motion_framerate)
        current_time = time.time()
        if current_time - self.last_saved_time < self.frame_interval:
            return
        
        try:
            # Update the saved time
            self.last_saved_time = current_time
            
            # Generate a timestamp string for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.output_folder, f"motion_{timestamp}.jpg")
            
            # Save the frame
            cv2.imwrite(filename, frame)
            print(f"Frame saved: {filename} (Motion FPS: {self.motion_framerate})")
        except Exception as e:
            print(f"Error saving frame: {e}")

    def setup_video_source(self, source):
        """Set up the video source (camera or file)."""
        try:
            # If source is an integer, it's a camera index
            if isinstance(source, int):
                # Flush the stream first
                temp_cap = cv2.VideoCapture(source)
                temp_cap.release()
                
                # Then start the capture
                self.cap = cv2.VideoCapture(source)
            # Otherwise, assume it's a file path
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Video file not found: {source}")
                self.cap = cv2.VideoCapture(source)
                
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")
                
        except Exception as e:
            print(f"Error setting up video source: {e}")
            sys.exit(1)

    def process_frame(self, frame):
        """Process a single frame for motion detection."""
        # Resize and convert to grayscale
        frame = imutils.resize(frame, width=750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize first_frame if needed
        if self.first_frame is None:
            self.first_frame = gray
            self.next_frame = gray
            return frame, False
        
        # Update reference frame if needed
        self.delay_counter += 1
        if self.delay_counter > self.frames_to_persist:
            self.delay_counter = 0
            self.first_frame = self.next_frame
        
        # Set next frame
        self.next_frame = gray
        
        # Calculate difference between frames
        frame_delta = cv2.absdiff(self.first_frame, self.next_frame)
        
        # Apply threshold using the configurable threshold value
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for motion in contours
        transient_movement = False
        for c in contours:
            # If contour is large enough, it's considered motion
            if cv2.contourArea(c) > self.min_size_for_movement:
                transient_movement = True
                (x, y, w, h) = cv2.boundingRect(c)
                # Draw rectangle around motion area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Handle consecutive motion frame counting
        if transient_movement:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0
        
        # Check if motion has persisted for enough consecutive frames
        significant_motion = self.consecutive_motion_frames >= self.minimum_motion_frames
        
        # Update motion persistence counter only if we have significant motion
        if significant_motion:
            self.movement_persistent_counter = self.movement_detected_persistence
        
        # Check if motion is still being detected
        motion_detected = self.movement_persistent_counter > 0
        
        # If motion is detected and frame saving is enabled, save the frame
        if motion_detected and self.save_frames:
            self._save_frame(frame)
        
        # Prepare visual feedback
        if motion_detected:
            text = f"Movement Detected {self.movement_persistent_counter}"
            self.movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"
        
        # Add text overlay with configuration information
        cv2.putText(frame, text, (10, 35), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Threshold: {self.threshold}", (10, 65), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Min Motion Frames: {self.minimum_motion_frames} (Current: {self.consecutive_motion_frames})", 
                   (10, 85), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add text indicating if frame saving is enabled and motion framerate
        if self.save_frames:
            cv2.putText(frame, f"Saving frames to: {self.output_folder} (FPS: {self.motion_framerate})", 
                       (10, 105), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert frame_delta to color for display
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        
        # Create display frame (combined original and delta)
        display_frame = np.hstack((frame_delta, frame))
        
        return display_frame, motion_detected

    def run(self):
        """Main loop for motion detection."""
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("End of video stream or error in capturing frame")
                    break
                
                # Process the frame
                display_frame, motion_detected = self.process_frame(frame)
                
                # Display the results
                cv2.imshow("Motion Detection", display_frame)
                
                # Check for quit command and parameter adjustments
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # Increase threshold with '+'
                elif key == ord('+') or key == ord('='):  # '=' is on same key as '+' without shift
                    self.threshold = min(100, self.threshold + 1)
                    print(f"Threshold increased to {self.threshold}")
                # Decrease threshold with '-'
                elif key == ord('-'):
                    self.threshold = max(1, self.threshold - 1)
                    print(f"Threshold decreased to {self.threshold}")
                # Increase minimum motion frames with '['
                elif key == ord('['):
                    self.minimum_motion_frames = max(1, self.minimum_motion_frames - 1)
                    print(f"Minimum motion frames decreased to {self.minimum_motion_frames}")
                # Decrease minimum motion frames with ']'
                elif key == ord(']'):
                    self.minimum_motion_frames += 1
                    print(f"Minimum motion frames increased to {self.minimum_motion_frames}")
                # Toggle frame saving with 's'
                elif key == ord('s'):
                    self.save_frames = not self.save_frames
                    if self.save_frames:
                        self._create_output_directory()
                        print(f"Frame saving enabled, saving to: {self.output_folder}")
                    else:
                        print("Frame saving disabled")
                # Increase motion framerate with 'up'
                elif key == ord('a'):  
                    self.motion_framerate = min(100, self.motion_framerate + 1)
                    self.frame_interval = 1.0 / max(2, self.motion_framerate)
                    print(f"Motion framerate increased to {self.motion_framerate}")
                # Decrease motion framerate with 'down'
                elif key == ord('d'):  
                    self.motion_framerate = max(2, self.motion_framerate - 1)
                    self.frame_interval = 1.0 / max(2, self.motion_framerate)
                    print(f"Motion framerate decreased to {self.motion_framerate}")
                
        except Exception as e:
            print(f"Error in motion detection: {e}")
        
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Motion Detection using OpenCV')
    
    # Add arguments
    parser.add_argument('-v', '--video', 
                        help='Path to video file (if not specified, webcam will be used)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('-f', '--frames-to-persist', type=int, default=10,
                        help='Number of frames to persist before comparing')
    parser.add_argument('-m', '--min-motion-size', type=int, default=2000,
                        help='Minimum size for a motion to be detected')
    parser.add_argument('-p', '--persistence', type=int, default=100,
                        help='How long motion is considered active after detection')
    parser.add_argument('-t', '--threshold', type=int, default=25,
                        help='Threshold for pixel change detection (1-100, default: 25)')
    parser.add_argument('-n', '--min-frames', type=int, default=1,
                        help='Minimum consecutive frames with motion before triggering detection (1-100, default: 1)')
    parser.add_argument('-s', '--save-frames', action='store_true',
                        help='Save frames when motion is detected (1 frame per second)')
    parser.add_argument('-o', '--output-folder', default='motion_detection',
                        help='Folder to save motion frames to (default: motion_detection)')
    parser.add_argument('-r', '--motion-framerate', type=int, default=15,
                        help='Framerate for motion capture (2-100, default: 15)')
    
    # Validate parameter ranges
    args = parser.parse_args()
    if args.threshold < 1 or args.threshold > 100:
        parser.error("Threshold must be between 1 and 100")
    if args.min_frames < 1:
        parser.error("Minimum motion frames must be at least 1")
    if args.motion_framerate < 2 or args.motion_framerate > 100:
        parser.error("Motion framerate must be between 2 and 100")
    
    return args


def main():
    """Main entry point of the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up configuration
    config = {
        'frames_to_persist': args.frames_to_persist,
        'min_size_for_movement': args.min_motion_size,
        'movement_detected_persistence': args.persistence,
        'threshold': args.threshold,
        'minimum_motion_frames': args.min_frames,
        'save_frames': args.save_frames,
        'output_folder': args.output_folder,
        'motion_framerate': args.motion_framerate,
    }
    
    # Determine video source
    if args.video:
        config['source'] = args.video
    else:
        config['source'] = args.camera
    
    # Create and run motion detector
    detector = MotionDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
