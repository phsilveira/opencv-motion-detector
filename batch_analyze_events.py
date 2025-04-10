#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/batch_analyze_events.py
import os
import json
import base64
import logging
import asyncio
import argparse
import glob
import yaml
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

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
logger = logging.getLogger("BatchAnalyzer")

class EventAnalyzer:
    def __init__(self, config):
        """Initialize the event analyzer with given configuration."""
        self.base_folder = config.get("base_folder", "motion_detected")
        self.max_images_to_analyze = config.get("max_images", 100)
        self.openai_api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4o")
        self.force_reanalyze = config.get("force_reanalyze", False)
        self.start_from = config.get("start_from", None)
        self.dry_run = config.get("dry_run", False)
        self.filter = config.get("filter", None)
        self.prompts_file = config.get("prompts_file", "auria_camera_prompts.yaml")
        
        # Initialize OpenAI client
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it in .env file or pass via --api-key")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Load camera prompts
        self.camera_prompts = self.load_camera_prompts()
        
        # Set up asyncio event loop
        self.loop = asyncio.get_event_loop()
    
    def load_camera_prompts(self) -> Dict[str, Any]:
        """Load camera prompts from YAML file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
            logger.info(f"Successfully loaded {len(prompts)} camera prompts from {self.prompts_file}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading camera prompts from {self.prompts_file}: {e}")
            return {}

    def find_event_folders(self) -> List[str]:
        """Find all event folders in the base directory."""
        event_folders = [f for f in glob.glob(os.path.join(self.base_folder, "event_*")) 
                        if os.path.isdir(f)]
        
        # Sort by folder name (which includes timestamp)
        event_folders.sort()
        
        # Filter if requested
        if self.filter:
            event_folders = [f for f in event_folders if self.filter in os.path.basename(f)]
        
        # Start from a specific folder if requested
        if self.start_from:
            start_idx = None
            for i, folder in enumerate(event_folders):
                if self.start_from in folder:
                    start_idx = i
                    break
            
            if start_idx is not None:
                logger.info(f"Starting from folder: {os.path.basename(event_folders[start_idx])}")
                event_folders = event_folders[start_idx:]
            else:
                logger.warning(f"Starting folder with '{self.start_from}' not found")
        
        return event_folders
    
    def needs_analysis(self, event_folder: str) -> bool:
        """Check if an event folder needs analysis."""
        analysis_file = os.path.join(event_folder, "analysis.txt")
        analysis_json = os.path.join(event_folder, "analysis.json")
        
        # If force reanalysis is enabled, always analyze
        if self.force_reanalyze:
            return True
        
        # If both files exist, no analysis needed
        if os.path.exists(analysis_file) and os.path.exists(analysis_json):
            return False
        
        return True
    
    def get_image_files(self, event_folder: str) -> List[str]:
        """Get image files from an event folder."""
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = []
        
        try:
            # Get all image files
            for file in os.listdir(event_folder):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(event_folder, file))
            
            # Sort by filename to maintain sequence
            # The files should start with sequence numbers like 0001_, 0002_, etc.
            image_files.sort()
            
            return image_files
        except Exception as e:
            logger.error(f"Error getting image files from {event_folder}: {e}")
            return []
    
    async def analyze_all_events(self):
        """Find and analyze all events that need analysis."""
        event_folders = self.find_event_folders()
        logger.info(f"Found {len(event_folders)} event folders")
        
        # Filter for events that need analysis
        events_to_analyze = [folder for folder in event_folders if self.needs_analysis(folder)]
        logger.info(f"{len(events_to_analyze)} events need analysis")
        
        if not events_to_analyze:
            logger.info("No events to analyze")
            return
        
        if self.dry_run:
            logger.info("Dry run mode - would analyze these folders:")
            for folder in events_to_analyze:
                logger.info(f"  {os.path.basename(folder)}")
            return
        
        # Process each event
        for folder in tqdm(events_to_analyze, desc="Analyzing events"):
            event_id = os.path.basename(folder)
            image_files = self.get_image_files(folder)
            
            if not image_files:
                logger.warning(f"No image files found in {event_id}")
                continue
            
            logger.info(f"Analyzing event {event_id} with {len(image_files)} images " +
                        f"(using max {self.max_images_to_analyze})")
            
            try:
                # Analyze the event
                await self.analyze_event(event_id, folder, image_files)
            except Exception as e:
                logger.error(f"Error analyzing event {event_id}: {e}")
    
    async def analyze_event(self, event_id: str, event_folder: str, image_files: List[str]):
        """Analyze a single event using the two-step process."""
        if not image_files:
            logger.warning(f"No images to analyze for event {event_id}")
            return
            
        # Limit the number of images to analyze
        images_to_analyze = image_files[:self.max_images_to_analyze]
        
        # Step 1: Classify camera scenario using the first image
        first_image = images_to_analyze[0]
        scenario_type = await self.classify_camera_scenario(first_image)
        logger.info(f"Classified event {event_id} as: {scenario_type}")
        
        # Step 2: Analyze the event using the appropriate prompt
        analysis = await self.analyze_with_scenario_prompt(scenario_type, event_id, images_to_analyze)
        
        # Save the analysis with the scenario classification
        self.save_analysis(event_id, event_folder, analysis, scenario_type)
    
    async def classify_camera_scenario(self, image_path: str) -> str:
        """
        Step 1: Classify the camera scenario type based on the first image.
        
        Returns one of:
        - camera_analyzer_prompt_external_street
        - camera_analyzer_prompt_internal_closed
        - camera_analyzer_prompt_internal_to_outside
        """
        try:
            # Encode the image to base64
            base64_image = self.encode_image(image_path)
            if not base64_image:
                logger.warning(f"Could not encode image {image_path}, using default scenario")
                return "camera_analyzer_prompt_external_street"  # Default
            
            # Classification prompt
            classification_prompt = """
            Classify this camera view into exactly one of the following three categories:
            
            1. external_street: Camera positioned externally, viewing a street, sidewalk, or outside area.
            2. internal_closed: Camera inside a closed environment without view to outside (rooms, halls, office).
            3. internal_to_outside: Camera inside looking toward outside areas like yards, gardens, or external corridors.
            
            Respond with ONLY the category name, nothing else.
            """
            
            # Create the API request
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": classification_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=50
            )
            
            # Process the response
            result = response.choices[0].message.content.strip().lower()
            logger.info(f"Camera classification result: {result}")
            
            # Map the classification to the appropriate prompt key
            if "external_street" in result:
                return "camera_analyzer_prompt_external_street"
            elif "internal_closed" in result:
                return "camera_analyzer_prompt_internal_closed"
            elif "internal_to_outside" in result:
                return "camera_analyzer_prompt_internal_to_outside"
            else:
                # Default fallback
                logger.warning(f"Could not determine camera type from response: {result}")
                return "camera_analyzer_prompt_external_street"
                
        except Exception as e:
            logger.error(f"Error in camera classification: {e}")
            return "camera_analyzer_prompt_external_street"  # Default fallback
    
    async def analyze_with_scenario_prompt(self, scenario_type: str, event_id: str, snapshot_files: List[str]) -> str:
        """
        Step 2: Analyze the images using the scenario-specific prompt from the YAML file.
        """
        # Get the appropriate prompt for the scenario
        if scenario_type not in self.camera_prompts:
            logger.warning(f"No prompt found for scenario {scenario_type}, using default system prompt")
            system_prompt = """You are the Camera Analyzer of the Vision Agent.
            Analyze the security camera images and provide an assessment of any potential security risks."""
        else:
            system_prompt = self.camera_prompts[scenario_type]["system_prompt"]
        
        logger.info(f"Using prompt for scenario: {scenario_type}")
        
        # Prepare image content for API
        content_list = []
        for image_path in snapshot_files:
            base64_image = self.encode_image(image_path)
            if base64_image:
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    }
                )
        
        if not content_list:
            return "No valid images to analyze"
        
        # Construct messages for API call
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list}
        ]
        
        try:
            logger.info(f"Sending analysis request to OpenAI API for event {event_id}")
            params = {
                "model": self.model,
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
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image to base64 for API transmission."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image {os.path.basename(image_path)}: {e}")
            return None
    
    def save_analysis(self, event_id: str, event_folder: str, analysis: str, scenario_type: str):
        """Save the analysis results for an event, including the scenario classification."""
        if not os.path.exists(event_folder):
            logger.warning(f"Event directory {event_folder} does not exist. Analysis not saved.")
            return
        
        result_data = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "scenario_type": scenario_type,
            "api_version": self.model
        }
        
        try:
            # Save text version
            with open(os.path.join(event_folder, "analysis.txt"), 'w') as f:
                f.write(analysis)
            
            # Save JSON version with additional metadata
            with open(os.path.join(event_folder, "analysis.json"), 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Analysis saved for event {event_id} (scenario: {scenario_type})")
        except Exception as e:
            logger.error(f"Error saving analysis for event {event_id}: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch analyze motion detection events')
    
    parser.add_argument('-i', '--input-folder', default='motion_detected',
                        help='Base folder containing event subfolders (default: motion_detected)')
    parser.add_argument('--api-key', 
                        help='OpenAI API key (default: read from OPENAI_API_KEY environment variable)')
    parser.add_argument('--max-images', type=int, default=100,
                        help='Maximum number of images to analyze per event (default: 100)')
    parser.add_argument('--model', default='gpt-4o',
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--force', action='store_true',
                        help='Force reanalysis of events even if they have existing analysis files')
    parser.add_argument('--start-from', 
                        help='Start analysis from a specific event ID (partial match)')
    parser.add_argument('--filter', 
                        help='Only analyze events with this string in their ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be analyzed without making API calls')
    parser.add_argument('--prompts-file', default='auria_camera_prompts.yaml',
                        help='YAML file containing camera prompts (default: auria_camera_prompts.yaml)')
    
    return parser.parse_args()

async def main():
    """Main entry point of the application."""
    args = parse_arguments()
    
    config = {
        'base_folder': args.input_folder,
        'max_images': args.max_images,
        'openai_api_key': args.api_key,
        'model': args.model,
        'force_reanalyze': args.force,
        'start_from': args.start_from,
        'dry_run': args.dry_run,
        'filter': args.filter,
        'prompts_file': args.prompts_file,
    }
    
    try:
        analyzer = EventAnalyzer(config)
        await analyzer.analyze_all_events()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
    finally:
        logger.info("Batch analysis complete")

if __name__ == "__main__":
    asyncio.run(main())