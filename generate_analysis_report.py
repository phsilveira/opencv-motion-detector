#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/generate_analysis_report.py
import os
import re
import pandas as pd
import glob
from datetime import datetime
import logging
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnalysisReport")

def load_camera_prompts(yaml_path="auria_camera_prompts.yaml"):
    """Load camera prompts from YAML file"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
        logger.info(f"Successfully loaded camera prompts from {yaml_path}")
        return prompts
    except Exception as e:
        logger.error(f"Error loading camera prompts from {yaml_path}: {e}")
        return {}

def classify_camera_scenario(first_image_path):
    """
    Use ChatGPT to classify the camera scenario type based on the first image.
    
    This is a placeholder function. In a real implementation, this would:
    1. Call the OpenAI API with the image
    2. Use a classification prompt to determine scenario type
    3. Return the appropriate prompt key
    """
    logger.info(f"Classifying camera scenario for image: {first_image_path}")
    return "camera_analyzer_prompt_external_street"

def analyze_event_with_prompt(images_folder, prompt_type, prompts_dict):
    """
    Apply the selected prompt to analyze the event using ChatGPT
    
    This is a placeholder function. In a real implementation, this would:
    1. Load the selected prompt from the prompts_dict
    2. Call the OpenAI API with the images and the prompt
    3. Return the analysis results
    """
    logger.info(f"Analyzing event with prompt type: {prompt_type}")
    return """
**Contexto**: Ambiente externo durante o dia com boa iluminação, voltado para uma rua residencial.

**Pessoas e veículos**: Foram observadas 2 pessoas caminhando na calçada e um veículo parado próximo.

**Comportamentos e sinais de risco**: Nenhum comportamento suspeito identificado.

**Label**: Normal

**Avaliação Geral de Risco**: Baixo - **Não**
"""

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
        image_files.sort()
        first_image_filename = image_files[0]
        return os.path.join(folder_path, first_image_filename), first_image_filename
    return None, None

def parse_image_filename(filename):
    """Parse the image filename to extract camera ID and event datetime"""
    if not filename:
        return None, None
    
    try:
        parts = filename.split('_')
        if len(parts) > 1:
            cam_id = parts[1]
        else:
            cam_id = None
            
        if len(parts) > 3:
            date_str = parts[2]
            time_str = parts[3]
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
            result = match.group(1).strip()
            return result
    
    return "N/A"

def parse_event_folder(event_folder):
    """Parse an event folder and extract relevant information"""
    folder_name = os.path.basename(event_folder)
    image_count = count_images(event_folder)
    first_image_path, first_image_filename = get_first_image(event_folder)
    
    cam_id, event_datetime = parse_image_filename(first_image_filename)
    scenario_type = classify_camera_scenario(first_image_path)
    
    analysis_file = os.path.join(event_folder, "analysis.txt")
    
    if not os.path.exists(analysis_file):
        logger.warning(f"No analysis.txt found in {folder_name}")
        
        prompts = load_camera_prompts()
        if prompts and scenario_type in prompts:
            logger.info(f"Would create analysis for {folder_name} using {scenario_type}")
        
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
            "avaliacao": "N/A",
            "scenario_type": scenario_type
        }
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
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
                "avaliacao": avaliacao,
                "scenario_type": scenario_type
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
            "avaliacao": "N/A",
            "scenario_type": scenario_type
        }

def is_dangerous_camera(cam_id):
    """Determine if the camera ID is for a dangerous event"""
    dangerous_labels = [
        'walk', 'lying', 'sit', 'hit', 'throw', 
        'sneak', 'fall', 'struggle', 'kick', 
        'grab', 'gun', 'run', 'videoplayback', 
        'altercation', 'burglary', 'climbing-wall',
        'break-in', 'car-break-in',  'robbery',
    ]
    
    if cam_id is None:
        return False
    
    return any(label.lower() in str(cam_id).lower() for label in dangerous_labels)

def is_predicted_dangerous(avaliacao):
    """Determine if the event is predicted as dangerous based on avaliacao"""
    if avaliacao is None or avaliacao == "N/A":
        return False
    
    return "**Sim**" in avaliacao

def generate_report(base_folder="motion_detected", output_file=None):
    """Generate a report of all event folders"""
    logger.info(f"Scanning folder: {base_folder}")
    
    event_folders = [f for f in glob.glob(os.path.join(base_folder, "event_*")) 
                     if os.path.isdir(f)]
    
    if not event_folders:
        logger.warning(f"No event folders found in {base_folder}")
        return
    
    logger.info(f"Found {len(event_folders)} event folders")
    
    results = []
    for folder in event_folders:
        logger.info(f"Processing folder: {os.path.basename(folder)}")
        data = parse_event_folder(folder)
        results.append(data)
    
    df = pd.DataFrame(results)
    
    df['real'] = df['cam_id'].apply(is_dangerous_camera)
    df['predict'] = df['avaliacao'].apply(is_predicted_dangerous)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_report_{timestamp}.csv"
    
    df.to_csv(output_file, index=False)
    logger.info(f"Report saved to {output_file}")
    
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False)
        logger.info(f"Excel report saved to {excel_file}")
    except Exception as e:
        logger.warning(f"Could not create Excel file: {e}")
    
    if len(df) > 0:
        try:
            y_true = np.array(df['real'])
            y_pred = np.array(df['predict'])
            
            cm = confusion_matrix(y_true, y_pred)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print("\n=== Confusion Matrix ===")
            print("              Predicted")
            print("              Not Dangerous  Dangerous")
            print(f"Actual Not Dangerous  {cm[0][0]:12d}  {cm[0][1]:9d}")
            print(f"      Dangerous       {cm[1][0]:12d}  {cm[1][1]:9d}")
            
            print("\n=== Classification Metrics ===")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {e}")
    
    print("\n=== Analysis Report Summary ===")
    print(f"Total events analyzed: {len(results)}")
    print("\nSample of the report (first 5 rows):")
    print(df[["folder_name", "image_count", "label", "avaliacao", "real", "predict", "scenario_type"]].head().to_string())
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
    
    df = generate_report(base_folder=args.input_folder, output_file=args.output_file)