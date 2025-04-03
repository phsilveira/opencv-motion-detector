#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/generate_analysis_report.py
import os
import re
import pandas as pd
import glob
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnalysisReport")

def count_images(folder_path):
    """Count image files in a folder"""
    image_extensions = ('.jpg', '.jpeg', '.png')
    return sum(1 for file in os.listdir(folder_path) 
               if file.lower().endswith(image_extensions))

def get_first_image(folder_path):
    """Get the path to the first image in a folder and its filename"""
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [file for file in os.listdir(folder_path)
                  if file.lower().endswith(image_extensions)]
    
    if image_files:
        # Sort to ensure consistent results
        image_files.sort()
        first_image_filename = image_files[0]
        return os.path.join(folder_path, first_image_filename), first_image_filename
    return None, None

def parse_image_filename(filename):
    """Parse the image filename to extract camera ID and event datetime
    Format example: 0001_6890_20250219_153816_frame_20250402_133726_000010.jpg
    """
    if not filename:
        return None, None
    
    try:
        # Split the filename by underscore
        parts = filename.split('_')
        
        # Extract camera ID (should be the second part)
        if len(parts) > 1:
            cam_id = parts[1]
        else:
            cam_id = None
            
        # Extract date and time (should be third and fourth parts)
        if len(parts) > 3:
            date_str = parts[2]
            time_str = parts[3]
            
            # Format: YYYYMMDD_HHMMSS
            if len(date_str) == 8 and len(time_str) == 6:
                event_datetime = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            else:
                event_datetime = None
        else:
            event_datetime = None
            
        return cam_id, event_datetime
    except Exception as e:
        logger.warning(f"Error parsing filename {filename}: {e}")
        return None, None

def extract_section(text, section_name):
    """Extract content of a specific section from analysis text"""
    # Define patterns for each section
    patterns = {
        "Contexto": r'\*\*Contexto\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
        "Pessoas e veículos": r'\*\*Pessoas e veículos\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
        "Comportamentos e sinais de risco": r'\*\*Comportamentos e sinais de risco\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
        "Label": r'\*\*Label\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
        "Avaliação Geral de Risco": r'\*\*Avaliação Geral de Risco\*\*:\s*(.*?)(?=\Z)'
    }
    
    if section_name in patterns:
        pattern = patterns[section_name]
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Clean up the extracted text (remove extra whitespace, newlines)
            result = match.group(1).strip()
            return result
    
    return "N/A"  # Return N/A if section not found

def parse_event_folder(event_folder):
    """Parse an event folder and extract relevant information"""
    folder_name = os.path.basename(event_folder)
    image_count = count_images(event_folder)
    first_image_path, first_image_filename = get_first_image(event_folder)
    
    # Parse the first image filename to extract metadata
    cam_id, event_datetime = parse_image_filename(first_image_filename)
    
    # Look for analysis.txt file
    analysis_file = os.path.join(event_folder, "analysis.txt")
    
    if not os.path.exists(analysis_file):
        logger.warning(f"No analysis.txt found in {folder_name}")
        return {
            "folder_name": folder_name,
            "image_count": image_count,
            "first_image": first_image_path,
            "cam_id": cam_id,
            "event_datetime": event_datetime,
            "contexto": "N/A",
            "pessoas": "N/A",
            "comportamentos": "N/A",
            "label": "N/A",
            "avaliacao": "N/A"
        }
    
    # Read the analysis file
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract each section
            contexto = extract_section(content, "Contexto")
            pessoas = extract_section(content, "Pessoas e veículos")
            comportamentos = extract_section(content, "Comportamentos e sinais de risco")
            label = extract_section(content, "Label")
            avaliacao = extract_section(content, "Avaliação Geral de Risco")
            
            return {
                "folder_name": folder_name,
                "image_count": image_count,
                "first_image": first_image_path,
                "cam_id": cam_id,
                "event_datetime": event_datetime,
                "contexto": contexto,
                "pessoas": pessoas,
                "comportamentos": comportamentos,
                "label": label,
                "avaliacao": avaliacao
            }
    except Exception as e:
        logger.error(f"Error processing {analysis_file}: {e}")
        return {
            "folder_name": folder_name,
            "image_count": image_count,
            "first_image": first_image_path,
            "cam_id": cam_id,
            "event_datetime": event_datetime,
            "contexto": f"ERROR: {str(e)}",
            "pessoas": "N/A",
            "comportamentos": "N/A",
            "label": "N/A",
            "avaliacao": "N/A"
        }

def generate_report(base_folder="motion_detected", output_file=None):
    """Generate a report of all event folders"""
    logger.info(f"Scanning folder: {base_folder}")
    
    # Find all event folders
    event_folders = [f for f in glob.glob(os.path.join(base_folder, "event_*")) 
                     if os.path.isdir(f)]
    
    if not event_folders:
        logger.warning(f"No event folders found in {base_folder}")
        return
    
    logger.info(f"Found {len(event_folders)} event folders")
    
    # Process each folder
    results = []
    for folder in event_folders:
        logger.info(f"Processing folder: {os.path.basename(folder)}")
        data = parse_event_folder(folder)
        results.append(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Determine output file name if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_report_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Report saved to {output_file}")
    
    # Save to Excel if pandas has openpyxl support
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False)
        logger.info(f"Excel report saved to {excel_file}")
    except Exception as e:
        logger.warning(f"Could not create Excel file: {e}")
    
    # Print a sample of the data
    print("\n=== Analysis Report Summary ===")
    print(f"Total events analyzed: {len(results)}")
    print("\nSample of the report (first 5 rows):")
    print(df[["folder_name", "image_count", "label", "avaliacao"]].head().to_string())
    print(f"\nFull report saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate an analysis report from motion detection events')
    parser.add_argument('-i', '--input-folder', default='motion_detected',
                      help='Base folder containing event subfolders (default: motion_detected)')
    parser.add_argument('-o', '--output-file', 
                      help='Output file name (default: analysis_report_TIMESTAMP.csv)')
    
    args = parser.parse_args()
    
    # Generate the report
    df = generate_report(base_folder=args.input_folder, output_file=args.output_file)