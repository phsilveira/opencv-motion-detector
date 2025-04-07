#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion Watcher Script

This script monitors a directory for new image files and applies motion
detection algorithms to detect movement between consecutive frames.

Usage:
    python motion_watcher.py [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [-t THRESHOLD] 
                            [-n MIN_FRAMES] [-p PERSISTENCE] [-e EVENT_GAP]
                            [--enable-analysis] [--api-key API_KEY]
"""

import argparse
import cv2
import os
import time
import shutil
import numpy as np
import sys
import queue
import uuid
import json
import base64
import logging
import asyncio
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from collections import deque

# Try to import OpenAI, but don't fail if it's not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not found. Event analysis will be disabled.")
    print("To enable, install with: pip install openai python-dotenv")

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MotionWatcher")


class MotionDetector:
    def __init__(self, config):
        """Initialize the motion detector with the given configuration."""
        # Configuration parameters
        self.threshold = config.get('threshold', 25)
        self.minimum_motion_frames = config.get('min_motion_frames', 1)
        self.motion_persistence = config.get('persistence', 5)
        self.min_area_percent = config.get('min_area_percent', 5.0)
        self.event_gap = config.get('event_gap', 60)  # In seconds
        
        # Event analysis configuration
        self.enable_analysis = config.get('enable_analysis', False)
        self.openai_api_key = config.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
        self.max_images_to_analyze = config.get('max_images', 100)
        
        # Initialize OpenAI client if analysis is enabled
        self.openai_client = None
        if self.enable_analysis:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not available. Event analysis will be disabled.")
                self.enable_analysis = False
            elif not self.openai_api_key:
                logger.warning("OpenAI API key not found. Event analysis will be disabled.")
                self.enable_analysis = False
            else:
                try:
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                    logger.info("OpenAI client initialized. Event analysis enabled.")
                except Exception as e:
                    logger.error(f"Error initializing OpenAI client: {e}")
                    self.enable_analysis = False
        
        # Initialize processing variables
        self.last_frame = None
        self.consecutive_motion_frames = 0
        self.motion_active = False
        self.motion_counter = 0
        self.processed_files = set()
        self.min_size_for_movement = 0
        self.frame_area = 0
        
        # Buffer to store recent frames for saving when motion is detected
        self.frame_buffer = deque(maxlen=max(10, self.minimum_motion_frames * 2))  # Store more frames than needed
        
        # Event tracking variables
        self.current_event_id = None
        self.last_motion_time = None
        self.event_frame_count = 0
        self.total_events = 0
        self.event_images = []  # Track images in the current event
        
        # For visualization
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Ensure output directory exists
        self.output_folder = config.get('output_folder', 'motion_detected')
        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                logger.info(f"Created output directory: {self.output_folder}")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            
        # Create a display frame for the main thread to show
        self.current_display_frame = None
        
        # Store the event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def calculate_area_threshold(self, frame):
        """Calculate the minimum area threshold based on frame dimensions and percentage."""
        height, width = frame.shape[:2]
        self.frame_area = height * width
        self.min_size_for_movement = int(self.frame_area * (self.min_area_percent / 100.0))
        logger.info(f"Frame dimensions: {width}x{height}, total area: {self.frame_area}")
        logger.info(f"Minimum motion size: {self.min_size_for_movement} pixels ({self.min_area_percent}% of frame)")

    def detect_motion(self, current_frame):
        """Detect motion between the current frame and the last frame."""
        # If this is the first frame, calculate area threshold and store it
        if self.last_frame is None:
            self.calculate_area_threshold(current_frame)
            self.last_frame = current_frame
            return False, None
        
        # Convert frames to grayscale and apply Gaussian blur
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
        
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.GaussianBlur(gray_last, (21, 21), 0)
        
        # Calculate difference between frames
        frame_delta = cv2.absdiff(gray_last, gray_current)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=10)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for motion in contours
        transient_movement = False
        motion_rects = []
        
        for c in contours:
            contour_area = cv2.contourArea(c)
            # If contour is large enough, consider it motion
            if contour_area > self.min_size_for_movement:
                transient_movement = True
                motion_rects.append(cv2.boundingRect(c))
        
        # Update consecutive motion frame count
        if transient_movement:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0
        
        # Check if motion has persisted for enough consecutive frames
        significant_motion = self.consecutive_motion_frames >= self.minimum_motion_frames
        
        # Update motion state
        if significant_motion:
            self.motion_counter = self.motion_persistence
            self.motion_active = True
            # Update last motion time for event tracking
            self.last_motion_time = datetime.now()
        elif self.motion_counter > 0:
            self.motion_counter -= 1
            self.motion_active = True
        else:
            self.motion_active = False
            
        # Update last frame for next comparison
        self.last_frame = current_frame
        
        return self.motion_active, motion_rects

    def check_event_status(self):
        """Check and update event status based on motion and event gap."""
        # If there's no event gap, always use the same event
        if self.event_gap <= 0:
            if self.current_event_id is None:
                self.current_event_id = self._generate_event_id()
                self.total_events += 1
                self.event_images = []  # Reset event images
                self._create_event_directory(self.current_event_id)
            return
            
        current_time = datetime.now()
        
        # Start a new event if motion is detected and no event is active
        if self.motion_active and self.current_event_id is None:
            self.current_event_id = self._generate_event_id()
            self.event_frame_count = 0
            self.total_events += 1
            self.event_images = []  # Reset event images
            logger.info(f"New event started: {self.current_event_id}")
            self._create_event_directory(self.current_event_id)
            
        # Check if the current event has expired due to lack of motion
        elif (self.current_event_id is not None and 
              self.last_motion_time is not None and 
              not self.motion_active):
            time_since_last_motion = (current_time - self.last_motion_time).total_seconds()
            if time_since_last_motion > self.event_gap:
                logger.info(f"Event {self.current_event_id} ended after {self.event_frame_count} frames "
                          f"(no motion for {time_since_last_motion:.1f}s)")
                
                # Get the event directory before resetting the event ID
                event_dir = self.get_event_directory()
                
                # Analyze the event if enabled
                if self.enable_analysis and self.event_images:
                    # Run the analysis in the event loop
                    try:
                        self.loop.run_until_complete(self.analyze_event(self.current_event_id, self.event_images))
                    except Exception as e:
                        logger.error(f"Error analyzing event: {e}")
                
                # Reset the event
                self.current_event_id = None
                self.event_images = []
                
    def _generate_event_id(self):
        """Generate a unique event ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"event_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    def _create_event_directory(self, event_id):
        """Create a directory for the current event."""
        event_dir = os.path.join(self.output_folder, event_id)
        try:
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
                logger.info(f"Created event directory: {event_dir}")
        except Exception as e:
            logger.error(f"Error creating event directory: {e}")

    def get_event_directory(self):
        """Get the directory for the current event, or the base output directory if no event is active."""
        if self.current_event_id is None or self.event_gap <= 0:
            return self.output_folder
        return os.path.join(self.output_folder, self.current_event_id)

    async def analyze_event(self, event_id: str, event_images: List[str]):
        """Analyze an event using OpenAI's image recognition capabilities."""
        if not self.enable_analysis or not self.openai_client:
            return
        
        logger.info(f"Analyzing event {event_id} with {len(event_images)} images...")
        
        # Limit the number of images to analyze
        images_to_analyze = event_images[:self.max_images_to_analyze]
        
        # Analyze the images
        analysis = await self.analyze_snapshots(event_id, images_to_analyze)
        
        # Save the analysis
        self.save_analysis(event_id, analysis)
    
    async def analyze_snapshots(self, event_id: str, snapshot_files: List[str]) -> str:
        """Analyze snapshots using OpenAI's vision capabilities."""
        if not self.openai_client:
            return "OpenAI client not initialized"
        
        base64_images = []
        for snapshot in snapshot_files:
            try:
                with open(snapshot, "rb") as f:
                    base64_images.append(base64.b64encode(f.read()).decode('utf-8'))
            except Exception as e:
                logger.error(f"Error reading image {snapshot}: {e}")
        
        if not base64_images:
            return "No valid images to analyze"
        
        content_list = []
        
        for base64_image in base64_images:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )
        
        prompt_messages = [
            {"role": "system", "content": """    Você é o Camera Analyzer da Vision Agent da Auria.
    Sua tarefa é analisar uma sequência contínua de imagens (cada imagem = 1 segundo) e avaliar riscos de segurança.
    O nome do arquivo de cada frame inicia com a ordem dele na sequência de imagens: 0001 indica o primeiro frame, 0002 indica o segundo etc. Analise os frames na sequência correta.
    Para cada sequência, reporte brevemente:
    1. **Contexto**: tipo do local (interno/externo), condições (dia/noite, iluminação).
    2. **Pessoas e veículos**: número, o que estão fazendo, evoluções claras das ações ao longo da sequência.
    3. **Comportamentos e sinais de risco**: invasões, acessos não autorizados, violência, comportamentos incomuns ou suspeitos, sinais de arrombamento, portas / janelas abertas.
    4. **Label da câmera**: identificação (se disponível).
    5. **Avaliação Geral de Risco**: **Sim/Não** risco + breve justificativa.
    Se imagens ilegíveis, informe "Sem visão". Se não houver nada a relatar nas perguntas 2, 3 e 4, informe "Nenhum".
    **Exemplo:**
    - Contexto: Externo, noite, estacionamento.
    - Pessoas e veículos: indivíduo aproximando-se lentamente, tenta abrir porta ao final.
    - Comportamentos e sinais de risco: tentativa clara de acesso indevido, pedaços possivelmente da porta quebrados no chão.
    - Label: Estacionamento 4.
    - Avaliação Geral de Risco: **Sim**, tentativa explícita de invasão."""},
            {"role": "user", "content": content_list}
        ]
        
        try:
            params = {
                "model": 'chatgpt-4o-latest',
                "messages": prompt_messages,
                "max_tokens": 2048,
                "temperature": 1.0,
                "top_p": 1.0,
            }
            response = self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API for event {event_id}: {e}")
            return f"Failed to analyze the snapshots: {str(e)}"
    
    def save_analysis(self, event_id: str, analysis: str):
        """Save the analysis results for an event."""
        event_dir = os.path.join(self.output_folder, event_id)
        if not os.path.exists(event_dir):
            logger.warning(f"Event directory {event_dir} does not exist. Analysis not saved.")
            return
        
        result_data = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "api_version": "gpt-4o"
        }
        
        try:
            # Save text version
            with open(os.path.join(event_dir, "analysis.txt"), 'w') as f:
                f.write(analysis)
            
            # Save JSON version
            with open(os.path.join(event_dir, "analysis.json"), 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Analysis saved for event {event_id}")
        except Exception as e:
            logger.error(f"Error saving analysis for event {event_id}: {e}")

    def process_image(self, image_path):
        """Process an image file for motion detection without displaying it directly."""
        try:
            # Skip if we've already processed this file
            if image_path in self.processed_files:
                return
            
            # Add to processed files
            self.processed_files.add(image_path)
            
            # Limit the size of processed_files to prevent memory issues
            if len(self.processed_files) > 1000:
                self.processed_files = set(list(self.processed_files)[-500:])
            
            # Read the image
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Error reading image: {image_path}")
                return
            
            # Store the frame and path in the buffer
            self.frame_buffer.append((image_path, frame))
            
            # Detect motion
            motion_detected, motion_rects = self.detect_motion(frame)
            
            # Create a copy for visualization
            display_frame = frame.copy()
            
            # Draw boxes around motion areas
            if motion_rects:
                for (x, y, w, h) in motion_rects:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check if this is the first frame that triggered motion detection
            first_motion_detected = (self.consecutive_motion_frames == self.minimum_motion_frames)
            
            # Update event status (before saving images)
            self.check_event_status()
            
            # Add text overlay
            if motion_detected:
                text = f"Motion Detected ({self.motion_counter})"
                color = (0, 0, 255)  # Red
                
                # If motion is active, save the current frame
                self._save_frame(image_path, frame)
                
                # If this is the first frame that triggered motion, also save buffered frames
                if first_motion_detected and len(self.frame_buffer) > 1:
                    # Save all frames in the buffer except the current one (which was just saved)
                    for i, (buffered_path, buffered_frame) in enumerate(list(self.frame_buffer)[:-1]):
                        self._save_frame(buffered_path, buffered_frame, prefix=f"pre{i+1}_")
            else:
                text = "No Motion"
                color = (0, 255, 0)  # Green
            
            cv2.putText(display_frame, text, (10, 30), self.font, 0.7, color, 2)
            cv2.putText(display_frame, f"Threshold: {self.threshold}", (10, 60), 
                        self.font, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Min Frames: {self.minimum_motion_frames} (Current: {self.consecutive_motion_frames})", 
                        (10, 80), self.font, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Min Area: {self.min_area_percent}% ({self.min_size_for_movement} px)", 
                        (10, 100), self.font, 0.6, (255, 255, 255), 1)
            
            # Add event information
            if self.current_event_id:
                time_in_event = "N/A"
                if self.last_motion_time:
                    seconds_since_motion = (datetime.now() - self.last_motion_time).total_seconds()
                    time_in_event = f"{seconds_since_motion:.1f}s ago"
                
                cv2.putText(display_frame, f"Event: {self.current_event_id}", 
                            (10, 120), self.font, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Event Frames: {self.event_frame_count} (Last motion: {time_in_event})", 
                            (10, 140), self.font, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Event Gap: {self.event_gap}s, Total Events: {self.total_events}", 
                            (10, 160), self.font, 0.6, (255, 255, 255), 1)
                
                # Show analysis status
                if self.enable_analysis:
                    cv2.putText(display_frame, f"Analysis: Enabled (max {self.max_images_to_analyze} images)", 
                                (10, 180), self.font, 0.6, (255, 255, 255), 1)
                else:
                    cv2.putText(display_frame, "Analysis: Disabled", 
                                (10, 180), self.font, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(display_frame, f"No active event (Gap: {self.event_gap}s, Total: {self.total_events})", 
                            (10, 120), self.font, 0.6, (255, 255, 255), 1)
                if self.enable_analysis:
                    cv2.putText(display_frame, "Analysis: Enabled (runs when events end)", 
                                (10, 140), self.font, 0.6, (255, 255, 255), 1)
            
            # Update the current display frame - will be shown by main thread
            self.current_display_frame = display_frame
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
    
    def _save_frame(self, image_path, frame, prefix=""):
        """Save a frame to the output directory."""
        try:
            filename = os.path.basename(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine output directory based on event status
            output_dir = self.get_event_directory()
            
            # Create output filename
            if self.current_event_id and self.event_gap > 0:
                self.event_frame_count += 1
                new_filename = f"{prefix}{self.event_frame_count:04d}_{filename}"
            else:
                new_filename = f"{prefix}motion_{timestamp}_{filename}"
            
            # Create the full output path
            output_path = os.path.join(output_dir, new_filename)
            
            # Save the file (either copy original or write new frame)
            if prefix == "":  # For current frame, copy the original
                shutil.copy2(image_path, output_path)
            else:  # For buffered frames, write the frame directly
                cv2.imwrite(output_path, frame)
                
            logger.info(f"Saved frame: {new_filename}")
            
            # Add to event images for analysis
            if self.current_event_id:
                self.event_images.append(output_path)
                
        except Exception as e:
            logger.error(f"Error saving frame {image_path}: {e}")

    def update_display(self):
        """Update the display with the current frame (call from main thread only)."""
        if self.current_display_frame is not None:
            try:
                cv2.imshow("Motion Detection", self.current_display_frame)
            except Exception as e:
                logger.error(f"Error displaying frame: {e}")

    def update_min_area_percent(self, new_percent):
        """Update the minimum area percentage and recalculate the threshold."""
        self.min_area_percent = new_percent
        if self.frame_area > 0:
            self.min_size_for_movement = int(self.frame_area * (self.min_area_percent / 100.0))
            logger.info(f"Updated minimum motion size: {self.min_size_for_movement} pixels ({self.min_area_percent}% of frame)")
            
    def update_event_gap(self, new_gap):
        """Update the event gap parameter."""
        old_gap = self.event_gap
        self.event_gap = new_gap
        logger.info(f"Event gap changed from {old_gap}s to {self.event_gap}s")
        
        # If disabling events, create a special "continuous" event
        if new_gap <= 0 and self.current_event_id is None:
            self.current_event_id = self._generate_event_id()
            self.total_events += 1
            self.event_images = []
            self._create_event_directory(self.current_event_id)
            logger.info(f"Continuous event mode activated: {self.current_event_id}")

    def toggle_analysis(self):
        """Toggle event analysis on/off."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. Cannot enable analysis.")
            return
            
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Cannot enable analysis.")
            return
            
        self.enable_analysis = not self.enable_analysis
        
        if self.enable_analysis:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("Event analysis enabled")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.enable_analysis = False
        else:
            logger.info("Event analysis disabled")


class ImageWatcher(FileSystemEventHandler):
    """Watch for new image files and process them."""
    
    def __init__(self, detector, input_folder):
        super().__init__()
        self.detector = detector
        self.input_folder = input_folder
        self.is_running = True
        self.file_queue = queue.Queue()  # Queue for new files
        
    def process_existing_files(self):
        """Process any existing files in the input folder."""
        logger.info(f"Processing existing files in {self.input_folder}...")
        for filename in sorted(os.listdir(self.input_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(self.input_folder, filename)
                self.file_queue.put(file_path)

    def on_created(self, event):
        """Handle file creation events."""
        if self.is_running and not event.is_directory:
            file_path = event.src_path
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.info(f"New file detected: {file_path}")
                self.file_queue.put(file_path)
                
    def process_queue(self):
        """Process files from the queue (call from main thread)."""
        try:
            while not self.file_queue.empty():
                file_path = self.file_queue.get(block=False)
                self.detector.process_image(file_path)
        except queue.Empty:
            pass  # Queue is empty, nothing to do


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Motion detection for image files')
    
    parser.add_argument('-i', '--input-folder', default='camera_frames',
                        help='Folder to watch for new image files (default: camera_frames)')
    parser.add_argument('-o', '--output-folder', default='motion_detected',
                        help='Folder to save motion detected frames (default: motion_detected)')
    parser.add_argument('-t', '--threshold', type=int, default=25,
                        help='Threshold for motion detection (1-100, default: 25)')
    parser.add_argument('-n', '--min-frames', type=int, default=1,
                        help='Minimum consecutive frames with motion (default: 1)')
    parser.add_argument('-p', '--persistence', type=int, default=5,
                        help='How many frames long motion is considered active (default: 5)')
    parser.add_argument('-m', '--min-area-percent', type=float, default=5.0,
                        help='Minimum size for motion area as percentage of frame (0.1-100, default: 5.0)')
    parser.add_argument('-e', '--event-gap', type=int, default=5,
                        help='Seconds of no motion that ends an event (0 = continuous, default: 5)')
    parser.add_argument('--enable-analysis', action='store_true',
                        help='Enable event analysis with OpenAI')
    parser.add_argument('--api-key', 
                        help='OpenAI API key (default: read from OPENAI_API_KEY environment variable)')
    parser.add_argument('--max-images', type=int, default=5,
                        help='Maximum number of images to analyze per event (default: 5)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.threshold < 1 or args.threshold > 100:
        parser.error("Threshold must be between 1 and 100")
    if args.min_frames < 1:
        parser.error("Minimum motion frames must be at least 1")
    if args.min_area_percent < 0.1 or args.min_area_percent > 100:
        parser.error("Minimum area percentage must be between 0.1 and 100")
    if args.event_gap < 0:
        args.event_gap = 0  # Convert negative values to continuous mode
        
    return args


def main():
    """Main entry point of the application."""
    args = parse_arguments()
    
    config = {
        'threshold': args.threshold,
        'min_motion_frames': args.min_frames,
        'persistence': args.persistence,
        'min_area_percent': args.min_area_percent,
        'event_gap': args.event_gap,
        'output_folder': args.output_folder,
        'enable_analysis': args.enable_analysis,
        'openai_api_key': args.api_key,
        'max_images': args.max_images,
    }
    
    detector = MotionDetector(config)
    
    # Verify the input directory exists
    if not os.path.isdir(args.input_folder):
        logger.warning(f"Input folder does not exist: {args.input_folder}")
        logger.info("Creating folder...")
        try:
            os.makedirs(args.input_folder)
        except Exception as e:
            logger.error(f"Error creating input folder: {e}")
            sys.exit(1)
    
    # Create and start the file system watcher
    event_handler = ImageWatcher(detector, args.input_folder)
    observer = Observer()
    observer.schedule(event_handler, args.input_folder, recursive=False)
    
    try:
        logger.info(f"Starting to watch folder: {args.input_folder}")
        logger.info(f"Motion detected images will be saved to: {args.output_folder}")
        if args.event_gap > 0:
            logger.info(f"Event mode: Events end after {args.event_gap}s without motion")
        else:
            logger.info("Event mode: Continuous (all motion in one event)")
            
        if detector.enable_analysis:
            logger.info("Event analysis: Enabled (will analyze events when they end)")
        else:
            logger.info("Event analysis: Disabled")
        
        logger.info("Press Ctrl+C to stop")
        
        # # Process existing files first
        # event_handler.process_existing_files()
        
        observer.start()
        
        # Main loop
        while True:
            time.sleep(0.1)
            key = cv2.waitKey(1) & 0xFF
            
            # Check for quit command
            if key == ord('q'):
                break
            
            # Threshold adjustments
            elif key == ord('+') or key == ord('='):
                detector.threshold = min(100, detector.threshold + 1)
                logger.info(f"Threshold increased to {detector.threshold}")
            elif key == ord('-'):
                detector.threshold = max(1, detector.threshold - 1)
                logger.info(f"Threshold decreased to {detector.threshold}")
                
            # Min motion frames adjustments
            elif key == ord(']'):
                detector.minimum_motion_frames += 1
                logger.info(f"Minimum motion frames increased to {detector.minimum_motion_frames}")
            elif key == ord('['):
                detector.minimum_motion_frames = max(1, detector.minimum_motion_frames - 1)
                logger.info(f"Minimum motion frames decreased to {detector.minimum_motion_frames}")
                
            # Persistence adjustments
            elif key == ord('p'):
                detector.motion_persistence += 10
                logger.info(f"Motion persistence increased to {detector.motion_persistence}")
            elif key == ord('o'):
                detector.motion_persistence = max(1, detector.motion_persistence - 10)
                logger.info(f"Motion persistence decreased to {detector.motion_persistence}")
                
            # Min area percentage adjustments
            elif key == ord('a'):
                detector.update_min_area_percent(max(0.1, detector.min_area_percent - 1))
                logger.info(f"Min area percentage decreased to {detector.min_area_percent:.1f}%")
            elif key == ord('s'):
                detector.update_min_area_percent(min(100, detector.min_area_percent + 1))
                logger.info(f"Min area percentage increased to {detector.min_area_percent:.1f}%")
                
            # Event gap adjustments
            elif key == ord('g'):
                if detector.event_gap > 0:
                    detector.update_event_gap(max(0, detector.event_gap - 5))
                else:
                    detector.update_event_gap(5)  # Switch from continuous to event mode
            elif key == ord('h'):
                detector.update_event_gap(detector.event_gap + 5)
                
            # Toggle continuous mode
            elif key == ord('c'):
                if detector.event_gap > 0:
                    detector.update_event_gap(0)  # Enable continuous mode
                else:
                    detector.update_event_gap(60)  # Restore default event gap
                
            # Process files from the queue
            event_handler.process_queue()
            
            # Update display
            detector.update_display()
    
    except KeyboardInterrupt:
        print("Stopping motion watcher...")
    except Exception as e:
        print(f"Error in motion watcher: {e}")
    finally:
        event_handler.is_running = False
        observer.stop()
        observer.join()
        cv2.destroyAllWindows()
        print("Motion watcher stopped")


if __name__ == "__main__":
    main()
