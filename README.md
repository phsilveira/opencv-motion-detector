# Two-Part Motion Detection System

This project consists of three separate Python scripts that work together to implement a motion detection system:

1. **Camera Capture Script (`camera_capture.py`)**: Captures frames from a camera, RTSP stream, or video file at 1fps and saves them to a directory
2. **Motion Watcher Script (`motion_watcher.py`)**: Monitors the saved images for motion detection
3. **Event Analyzer Script (`event_analyzer.py`)**: Uses AI to analyze detected motion events and describe what's happening

## Camera Capture Script

The camera capture script continuously captures frames from a camera source, RTSP stream, or video file and saves them to a specified directory at a rate of 1 frame per second.

### Usage

```bash
python camera_capture.py [[-c CAMERA_INDEX] | [-r RTSP_URL] | [-v VIDEO_FILE]] [-o OUTPUT_FOLDER] [-s PLAYBACK_SPEED]
```

### Command Line Options

- `-c, --camera`: Camera device index
- `-r, --rtsp`: RTSP URL (default: rtsp://admin:rad001877@rc-uchida.dyndns.org:554/Streaming/Channels/102/)
- `-v, --video`: Path to a video file
- `-o, --output-folder`: Folder to save captured frames to (default: camera_frames)
- `-s, --speed`: Playback speed multiplier for video files (default: 1.0)

Note: You must use only one of `-c`, `-r`, or `-v` to specify the input source.

### Examples

Using the default RTSP stream:
```bash
python camera_capture.py
```

Using a specific camera device:
```bash
python camera_capture.py --camera 1 --output-folder my_camera_frames
```

Using a video file:
```bash
python camera_capture.py --video path/to/video.mp4 --output-folder video_frames --speed 2.0
```

test in batch
```bash
python run_tests.py -v "test_occurrences/video/67caf76edba1563e9204c48e/*.mp4" -m 2 --fps 1 -p 5
```

### Runtime Controls

While the camera capture is running with a video file:
- Press `q` to quit
- Press `+` to increase playback speed
- Press `-` to decrease playback speed

## Motion Watcher Script

The motion watcher script monitors a directory for new image files and applies motion detection algorithms to detect movement between consecutive frames.

### Usage

```bash
python motion_watcher.py [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [-t THRESHOLD] [-n MIN_FRAMES] [-p PERSISTENCE] [-m MIN_AREA_PERCENT] [-e EVENT_GAP]
```

### Command Line Options

- `-i, --input-folder`: Folder to watch for new image files (default: camera_frames)
- `-o, --output-folder`: Folder to save motion detected frames (default: motion_detected)
- `-t, --threshold`: Threshold for motion detection (1-100, default: 25)
- `-n, --min-frames`: Minimum consecutive frames with motion (default: 1)
- `-p, --persistence`: How long motion is considered active (default: 5)
- `-m, --min-area-percent`: Minimum size for motion area as percentage of frame (0.1-100, default: 5.0)
- `-e, --event-gap`: Seconds of no motion that ends an event (0 = continuous, default: 60)

### Example

```bash
python motion_watcher.py --input-folder my_camera_frames --threshold 30 --min-frames 2 --event-gap 30
```

### Runtime Controls

While the motion watcher is running:
- Press `q` to quit
- Press `+` to increase detection threshold (less sensitive)
- Press `-` to decrease detection threshold (more sensitive)
- Press `]` to increase minimum motion frames
- Press `[` to decrease minimum motion frames
- Press `p` to increase motion persistence
- Press `o` to decrease motion persistence
- Press `a` to decrease minimum area percentage
- Press `s` to increase minimum area percentage
- Press `g` to decrease event gap time
- Press `h` to increase event gap time
- Press `c` to toggle continuous event mode (all motion in one event)

## Event Analyzer Script

The event analyzer script processes motion events captured by the motion watcher and uses OpenAI's GPT-4o model to generate descriptions of what triggered the motion detection.

### Prerequisites

1. You need an OpenAI API key to use this script
2. Export your API key as an environment variable or use the `-k` option:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file in the project directory with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### Usage

```bash
python event_analyzer.py [-i INPUT_FOLDER] [-l LIMIT] [-m MAX_IMAGES] [-k API_KEY]
```

### Command Line Options

- `-i, --input-folder`: Folder containing motion events (default: motion_detected)
- `-l, --limit`: Limit the number of events to analyze (default: 0 = analyze all)
- `-m, --max-images`: Maximum images to analyze per event (default: 5)
- `-k, --api-key`: OpenAI API key (default: read from OPENAI_API_KEY environment variable)

### Example

```bash
python event_analyzer.py --input-folder motion_detected --max-images 3
```

### Output

For each analyzed event, the script creates two files in the event directory:
- `analysis.txt`: Plain text description of what was detected in the event
- `analysis.json`: JSON file with the description and additional metadata

The script also keeps track of analyzed events, so it won't analyze the same event twice.

## Configuration Parameters

### Threshold

- **Type**: Integer
- **Range**: 1 - 100
- **Default**: 25

Threshold for declaring motion. The threshold determines how much pixel change is required to detect motion. Lower values make the detector more sensitive to subtle movements, while higher values require more substantial changes to trigger detection.

### Minimum Motion Frames

- **Type**: Integer
- **Range**: 1 - 1000s
- **Default**: 1

Picture frames must contain motion at least the specified number of frames in a row before they are detected as true motion. At the default of 1, all motion is detected. This feature helps filter out false detections when light conditions change or the camera adjusts sensitivity.

Recommended to keep this value between 1-5 for best results. Higher values may cause the detector to miss brief motion events.

### Persistence

- **Type**: Integer
- **Default**: 5

How many frames motion is considered active after initial detection, creating a "motion memory" effect.

### Minimum Area Percentage

- **Type**: Float
- **Range**: 0.1 - 100
- **Default**: 1.0

The minimum size of a motion area as a percentage of the total frame area. This helps filter out small movements or noise. For example, with the default value of 1.0%, a motion region must occupy at least 1% of the total frame area to be considered significant motion.

### Event Gap

- **Type**: Integer
- **Range**: 0 - 2147483647
- **Default**: 60

The number of seconds of no motion detected that triggers the end of an event. An event is defined as a series of motion images taken within a short timeframe. When motion is detected, a new event starts (unless one is already active). The event continues as long as motion is detected within the specified gap time. Once no motion is detected for the duration of the event gap, the event ends.

Setting this to 0 enables "continuous mode" where all motion is captured as part of the same event. This is useful for creating sequential frame numbering regardless of gaps between motion.

Each event creates a separate subdirectory in the output folder, helping to organize detected motion by distinct occurrences.

## How to Run the Complete System

1. Start the camera capture in one terminal:
```bash
python camera_capture.py
```

2. Start the motion watcher in another terminal:
```bash
python motion_watcher.py
```

3. Periodically run the event analyzer to interpret the detected events:
```bash
python event_analyzer.py
```

You can also set up the event analyzer to run automatically at regular intervals using a cron job or scheduled task.

3. The camera capture will continuously save images to the "camera_frames" folder
4. The motion watcher will detect motion in these images and copy frames with motion to the "motion_detected" folder

You can run both scripts with custom parameters to suit your environment and needs.


Scans the motion_detected directory for event folders
Identifies which events don't have analysis files (or processes all if --force is used)
For each event folder:
Collects the image files and sorts them by sequence
Sends them to OpenAI's vision API with the same prompt used in motion_watcher.py
Saves the analysis results in both text and JSON format
Features:

Command-line options to customize behavior:

--input-folder: Specify a different base folder
--max-images: Limit how many images to analyze per event (default: 20)
--model: Choose the OpenAI model (default: chatgpt-4o-latest)
--force: Reanalyze events even if they already have analysis files
--start-from: Start analysis from a specific event ID
--filter: Only analyze events with a specific string in their ID
--dry-run: Show what would be analyzed without making API calls
--api-key: Provide OpenAI API key (otherwise reads from environment)
Progress tracking using tqdm

Error handling for API failures or image loading issues

Logging with detailed information about the analysis process

To use the script:

# Basic usage (reads API key from environment variable)
python batch_analyze_events.py

# Specify a different folder and more options
python batch_analyze_events.py -i /path/to/events --max-images 10 --force

# Start from a specific event and only process a subset
python batch_analyze_events.py --start-from event_20250402 --filter 20250402

The script requires the following Python packages:

openai
python-dotenv
tqdm

python yolo_detector.py --input motion_detected/ --output-dir output/ --recursive --preserve-structure --debug