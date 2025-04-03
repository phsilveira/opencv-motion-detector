#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/extract_video_metadata.py
import os
import csv
import glob
import re
import cv2
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoMetadataExtractor")

def extract_video_duration(video_path):
    """Extract the duration of a video file in seconds"""
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not video.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        if fps > 0:
            duration_seconds = frame_count / fps
            video.release()
            return int(duration_seconds)
        else:
            logger.warning(f"Invalid FPS value for {video_path}")
            video.release()
            return None
    except Exception as e:
        logger.error(f"Error extracting duration from {video_path}: {e}")
        return None

def parse_filename(filename):
    """Parse the video filename to extract metadata
    Format example: 6883_20250219_153816.mp4
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    try:
        # Split by underscore
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            cam_id = parts[0]
            
            # Parse date and time
            date_str = parts[1]
            time_str = parts[2]
            
            if len(date_str) == 8 and len(time_str) == 6:
                record_timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            else:
                record_timestamp = None
                
            return cam_id, record_timestamp
        else:
            return None, None
    except Exception as e:
        logger.warning(f"Error parsing filename {filename}: {e}")
        return None, None

def extract_metadata(base_folder="test_occurrences/video", output_file="video_metadata.csv"):
    """Extract metadata from video files and save to CSV"""
    logger.info(f"Scanning folder: {base_folder}")
    
    # Find all occurrence folders
    occurrence_folders = [f for f in glob.glob(os.path.join(base_folder, "*")) 
                         if os.path.isdir(f)]
    
    if not occurrence_folders:
        logger.warning(f"No occurrence folders found in {base_folder}")
        return
    
    logger.info(f"Found {len(occurrence_folders)} occurrence folders")
    
    # Prepare data collection
    results = []
    
    # Process each occurrence folder
    for folder in occurrence_folders:
        occurrence_id = os.path.basename(folder)
        logger.info(f"Processing occurrence: {occurrence_id}")
        
        # Find all video files in this folder
        video_pattern = os.path.join(folder, "*.mp4")
        video_files = glob.glob(video_pattern)
        
        if not video_files:
            logger.warning(f"No video files found in {folder}")
            continue
            
        # Process each video file with a progress bar
        for video_path in tqdm(video_files, desc=f"Videos in {occurrence_id}"):
            filename = os.path.basename(video_path)
            
            # Extract metadata from filename
            cam_id, record_timestamp = parse_filename(filename)
            
            # Extract video duration
            duration_seconds = extract_video_duration(video_path)
            
            # Store the data
            results.append({
                'occurrence_id': occurrence_id,
                'filename': filename,
                'video_duration_in_seconds': duration_seconds,
                'cam_id': cam_id,
                'record_timestamp': record_timestamp,
                'full_path': video_path
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Metadata saved to {output_file}")
    
    # Save to Excel if possible
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False)
        logger.info(f"Excel report saved to {excel_file}")
    except Exception as e:
        logger.warning(f"Could not create Excel file: {e}")
    
    # Print a summary
    print("\n=== Video Metadata Summary ===")
    print(f"Total occurrences: {len(occurrence_folders)}")
    print(f"Total video files: {len(results)}")
    print("\nSample of the data (first 5 rows):")
    print(df[["occurrence_id", "filename", "cam_id", "video_duration_in_seconds"]].head().to_string())
    print(f"\nFull metadata saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract metadata from video files')
    parser.add_argument('-i', '--input-folder', default='test_occurrences/video',
                      help='Base folder containing occurrence subfolders (default: test_occurrences/video)')
    parser.add_argument('-o', '--output-file', default='video_metadata.csv',
                      help='Output CSV file name (default: video_metadata.csv)')
    
    args = parser.parse_args()
    
    # Extract metadata
    df = extract_metadata(base_folder=args.input_folder, output_file=args.output_file)