import cv2
import os
import argparse
import uuid
from datetime import datetime
import shutil

def normalize_video_name(video_path, category):
    """
    Rename the video to follow the pattern: {cam_id}_{YYYYMMDD}_{HHmmSS}.mp4
    
    Args:
        video_path: Path to the original video file
        category: The category/folder name to use as cam_id
        
    Returns:
        Path to the renamed video file
    """
    # Get the file extension
    _, ext = os.path.splitext(video_path)
    
    # Generate timestamp for the new filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create new filename with pattern: {cam_id}_{YYYYMMDD}_{HHmmSS}.ext
    new_filename = f"{category}_{timestamp}{ext}"
    
    # Determine the directory of the original file
    directory = os.path.dirname(video_path)
    
    # Create full path for the new file
    new_path = os.path.join(directory, new_filename)
    
    # Make a copy of the file with the new name
    # Using copy instead of rename to preserve the original
    print(f"Normalizing video name: {os.path.basename(video_path)} -> {new_filename}")
    shutil.copy2(video_path, new_path)
    
    return new_path

def video_to_images(video_path, output_fps=None, base_output_dir=None, category=None):
    """
    Convert a video file to a sequence of images
    
    Args:
        video_path: Path to the video file
        output_fps: If specified, extract frames at this rate. If None, extract all frames.
        base_output_dir: Base directory for outputs
        category: Video category name for organizing output
    """
    # Check if video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    
    # Create output directory with pattern consistent with other scripts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4()).split('-')[0]  # Using first part of UUID for brevity
    
    # Get video filename without extension to use as source name
    source_name = os.path.splitext(os.path.basename(video_path))[0]
    folder_name = f"event_{timestamp}_{unique_id}"
    
    # Create output directory structure - directly under base_output_dir without category subfolder
    if base_output_dir is None:
        base_output_dir = "motion_detected"
    
    output_dir = os.path.join(base_output_dir, folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frames to extract
    if output_fps is None or output_fps >= original_fps:
        # Extract all frames if output_fps is not specified or greater than original
        frames_to_extract = range(total_frames)
        actual_fps = original_fps
    else:
        # Calculate frame extraction interval
        interval = int(original_fps / output_fps)
        frames_to_extract = range(0, total_frames, interval)
        actual_fps = original_fps / interval
    
    print(f"Converting video: {video_path}")
    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {output_fps if output_fps else 'All frames'}")
    print(f"Actual output FPS: {actual_fps}")
    print(f"Saving frames to: {output_dir}")
    
    # Extract and save frames
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frames_to_extract:
            # Save the frame with pattern similar to camera_capture and motion_watcher
            # Format: NNNN_source_name_frame.jpg
            output_filename = f"{saved_count:04d}_{source_name}_frame.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        
        # Display progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    print(f"Extraction complete. Saved {saved_count} frames to {output_dir}")
    return output_dir

def process_videos_from_directory(base_dir, output_fps=None, base_output_dir="motion_detected"):
    """
    Process all videos from all categories in the specified directory
    
    Args:
        base_dir: Base directory containing category folders with videos
        output_fps: FPS for extracting frames
        base_output_dir: Base directory for outputs
    """
    if not os.path.isdir(base_dir):
        print(f"Error: Directory {base_dir} not found")
        return
    
    # Create the base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get all category folders
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not categories:
        print(f"No category folders found in {base_dir}")
        return
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    
    # Process videos in each category
    for category in categories:
        category_path = os.path.join(base_dir, category)
        print(f"\nProcessing category: {category}")
        
        # Get all video files in the category folder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        videos = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)) and
                  any(f.lower().endswith(ext) for ext in video_extensions)]
        
        if not videos:
            print(f"No videos found in category {category}")
            continue
            
        print(f"Found {len(videos)} videos in category {category}")
        
        # Process each video
        for video in videos:  # Limit to first 5 videos for demonstration
            original_video_path = os.path.join(category_path, video)
            
            # Normalize the video name to follow the pattern
            normalized_video_path = normalize_video_name(original_video_path, category)
            
            print(f"\nProcessing video: {os.path.basename(normalized_video_path)}")
            video_to_images(normalized_video_path, output_fps, base_output_dir, category)
            
            # Clean up the temporary normalized video if it's different from original
            if normalized_video_path != original_video_path and os.path.exists(normalized_video_path):
                os.remove(normalized_video_path)
                print(f"Removed temporary normalized video: {os.path.basename(normalized_video_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to sequence of images")
    parser.add_argument("--video_path", help="Path to a single video file")
    parser.add_argument("--videos_dir", help="Path to directory containing category folders with videos")
    parser.add_argument("--fps", type=float, help="Output frames per second (default: extract all frames)", default=1)
    parser.add_argument("--output_dir", help="Base directory for output", default="motion_detected")
    
    args = parser.parse_args()
    
    if args.videos_dir:
        process_videos_from_directory(args.videos_dir, args.fps, args.output_dir)
    elif args.video_path:
        video_to_images(args.video_path, args.fps, args.output_dir)
    else:
        print("Error: Either --video_path or --videos_dir must be specified")
        parser.print_help()