import os
import subprocess
import json
from datetime import timedelta

def get_video_duration(video_path):
    """Gets the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = json.loads(result.stdout).get("format", {}).get("duration", 0)
        return float(duration)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0

def format_duration(seconds):
    """Formats seconds into a readable time format (HH:MM:SS)."""
    return str(timedelta(seconds=seconds)).split('.')[0]

def get_folder_durations(root_folder):
    """Calculates the total duration of videos in each subfolder."""
    folder_details = {}
    
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        if os.path.isdir(folder_path):
            folder_total = 0
            videos_info = []
            
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(".mp4"):
                    video_path = os.path.join(folder_path, file_name)
                    duration = get_video_duration(video_path)
                    folder_total += duration
                    videos_info.append((file_name, duration))
            
            folder_details[folder_name] = {
                "total_duration": folder_total,
                "videos": videos_info
            }
    
    return folder_details

if __name__ == "__main__":
    root_folder = "./"  # Change this to your base directory
    folder_details = get_folder_durations(root_folder)
    
    grand_total = 0
    
    print("\n=== VIDEO DURATION ANALYSIS ===\n")
    
    for folder, details in folder_details.items():
        folder_total = details["total_duration"]
        grand_total += folder_total
        
        print(f"\n📁 FOLDER: {folder}")
        print(f"   Total Duration: {format_duration(folder_total)} ({folder_total:.2f} seconds)")
        print("   Videos:")
        
        for video_name, video_duration in details["videos"]:
            print(f"   - {video_name}: {format_duration(video_duration)} ({video_duration:.2f} seconds)")
    
    print("\n=== SUMMARY ===")
    print(f"Total Duration of All Videos: {format_duration(grand_total)} ({grand_total:.2f} seconds)")
    print(f"Number of Folders Processed: {len(folder_details)}")
