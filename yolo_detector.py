#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Detector

This script processes images with a YOLO model and saves detection results as JSON files.
Works with both individual image files and directories of images.

Usage:
    python yolo_detector.py --input path/to/image.jpg --model yolov8n.pt
    python yolo_detector.py --input path/to/images/directory --model yolov8n.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Union, Optional
import logging

try:
    from ultralytics import YOLO
    import numpy as np
    import cv2
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install required packages with:")
    print("pip install ultralytics opencv-python numpy tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLODetector")

class YOLODetector:
    """YOLO model wrapper for image detection and JSON output."""
    
    def __init__(self, model_path: str, confidence: float = 0.25):
        """Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model file
            confidence: Confidence threshold for detections (0-1)
        """
        logger.info(f"Loading YOLO model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.confidence = confidence
            logger.info(f"Model loaded successfully with confidence threshold: {confidence}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image_path: str) -> Dict:
        """Run detection on an image and return results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Read image with OpenCV to verify it's valid
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return {"error": "Failed to read image"}
            
            # Run inference
            results = self.model(img, conf=self.confidence)
            
            # Extract results
            return self._format_results(results[0], image_path)
        
        except Exception as e:
            logger.error(f"Error detecting objects in {image_path}: {e}")
            return {"error": str(e)}
    
    def _format_results(self, results, image_path: str) -> Dict:
        """Format YOLO results into a clean dictionary.
        
        Args:
            results: Results from YOLO model
            image_path: Path to the source image
            
        Returns:
            Dictionary with formatted results
        """
        # Get image dimensions
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        except:
            height, width = 0, 0
        
        # Extract boxes, confidence scores, and class names
        boxes = results.boxes
        detections = []
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get confidence and class
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = results.names[class_id]
            
            # Create detection object
            detection = {
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                }
            }
            detections.append(detection)
        
        # Create final result object
        return {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "image_width": width,
            "image_height": height,
            "timestamp": None,  # Can be filled in by caller if needed
            "detections": detections,
            "detection_count": len(detections)
        }
    
    def draw_debug_image(self, image_path: str, results: Dict, output_dir: Optional[str] = None) -> str:
        """Draw bounding boxes on image and save it.
        
        Args:
            image_path: Path to the original image
            results: Detection results dictionary
            output_dir: Optional directory to save debug image to
            
        Returns:
            Path to the saved debug image
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image for debug output: {image_path}")
                return None
            
            # Draw each detection
            for detection in results["detections"]:
                # Extract bbox coordinates
                bbox = detection["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                
                # Determine color based on class (for variety)
                class_id = detection["class_id"]
                np.random.seed(class_id)  # For consistent colors per class
                color = tuple(map(int, np.random.randint(0, 255, size=3).tolist()))
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with class name and confidence
                label = f"{detection['class']} {detection['confidence']:.2f}"
                
                # Get text size for background rectangle
                font_scale = 0.6
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    img, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    img, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255), 
                    font_thickness
                )
            
            # Determine where to save the debug image
            if output_dir:
                # Save in the specified output directory
                image_filename = os.path.basename(image_path)
                debug_image_path = os.path.join(output_dir, image_filename)
            else:
                # Create debug directory within the original image directory
                original_dir = os.path.dirname(image_path)
                debug_dir = os.path.join(original_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save the debug image
                image_filename = os.path.basename(image_path)
                debug_image_path = os.path.join(debug_dir, image_filename)
            
            cv2.imwrite(debug_image_path, img)
            logger.info(f"Saved debug image to {debug_image_path}")
            
            return debug_image_path
        
        except Exception as e:
            logger.error(f"Error creating debug image: {e}")
            return None
    
    def save_json(self, results: Dict, output_dir: Optional[str] = None) -> str:
        """Save detection results to a JSON file.
        
        Args:
            results: Detection results dictionary
            output_dir: Optional directory to save JSON to (uses image dir if None)
            
        Returns:
            Path to the saved JSON file
        """
        try:
            # Extract image path and create JSON path
            image_path = results["path"]
            image_dir = os.path.dirname(image_path)
            image_filename = os.path.basename(image_path)
            base_name = os.path.splitext(image_filename)[0]
            
            # Determine output directory
            save_dir = output_dir if output_dir else image_dir
            
            # Ensure output directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Create JSON path
            json_path = os.path.join(save_dir, f"{base_name}.json")
            
            # Save JSON
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved detection results to {json_path}")
            return json_path
        
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return None
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None, debug: bool = False) -> Dict:
        """Process a single image and save results as JSON.
        
        Args:
            image_path: Path to image file
            output_dir: Optional directory to save JSON to
            debug: Whether to generate debug images with bounding boxes
            
        Returns:
            Detection results dictionary
        """
        results = self.detect(image_path)
        if "error" not in results:
            self.save_json(results, output_dir)
            if debug:
                self.draw_debug_image(image_path, results, output_dir)
        return results
    
    def process_directory(self, directory: str, output_dir: Optional[str] = None, debug: bool = False, 
                          recursive: bool = False, preserve_structure: bool = False) -> List[Dict]:
        """Process all images in a directory.
        
        Args:
            directory: Directory containing images
            output_dir: Optional directory to save JSON files to
            debug: Whether to generate debug images with bounding boxes
            recursive: Whether to search for images in subdirectories recursively
            preserve_structure: Whether to preserve the directory structure in output
            
        Returns:
            List of detection results dictionaries
        """
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        if recursive:
            # Walk through all subdirectories
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            # Just get files in the main directory
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, file))
        
        if not image_files:
            logger.warning(f"No image files found in {directory}")
            return []
        
        # Process each image
        logger.info(f"Processing {len(image_files)} images in {directory}{' and subdirectories' if recursive else ''}")
        results = []
        
        for image_path in tqdm(image_files, desc="Processing images"):
            if preserve_structure and output_dir:
                # Get relative path to maintain directory structure
                rel_path = os.path.relpath(os.path.dirname(image_path), directory)
                # Create specific output directory for this file
                specific_output_dir = os.path.join(output_dir, rel_path) if rel_path != '.' else output_dir
                os.makedirs(specific_output_dir, exist_ok=True)
            else:
                specific_output_dir = output_dir
                
            result = self.process_image(image_path, specific_output_dir, debug)
            results.append(result)
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLO image detection with JSON output')
    
    parser.add_argument('-i', '--input', required=True, 
                      help='Path to an image file or directory of images')
    parser.add_argument('-m', '--model', default='yolov8n.pt',
                      help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('-o', '--output-dir', 
                      help='Directory to save JSON results (default: same as input)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                      help='Confidence threshold (0-1, default: 0.25)')
    parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug mode to output images with bounding boxes')
    parser.add_argument('-r', '--recursive', action='store_true',
                      help='Process images in subdirectories recursively')
    parser.add_argument('-p', '--preserve-structure', action='store_true',
                      help='Preserve directory structure in output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        # Initialize detector
        detector = YOLODetector(args.model, args.confidence)
        
        # Process input (file or directory)
        if os.path.isfile(args.input):
            logger.info(f"Processing single image: {args.input}")
            detector.process_image(args.input, args.output_dir, args.debug)
        
        elif os.path.isdir(args.input):
            logger.info(f"Processing directory: {args.input}")
            detector.process_directory(
                args.input, 
                args.output_dir, 
                args.debug, 
                args.recursive, 
                args.preserve_structure
            )
        
        else:
            logger.error(f"Input path does not exist: {args.input}")
            sys.exit(1)
        
        logger.info("Processing complete")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()