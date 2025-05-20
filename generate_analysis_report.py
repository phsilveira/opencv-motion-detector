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
import json
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

def extract_section(text, section_name, format_type="default"):
    """Extract content of a specific section from analysis text"""
    patterns = {
        "default": {
            "Contexto": r'\*\*Contexto\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Pessoas e veículos": r'\*\*Pessoas e veículos\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Comportamentos e sinais de risco": r'\*\*Comportamentos e sinais de risco\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Label": r'\*\*Label\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Avaliação Geral de Risco": r'\*\*Avaliação Geral de Risco\*\*:\s*(.*?)(?=\Z)'
        },
        "detailed": {
            "Contexto": r'-\s*\*\*Contexto\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Observações": r'-\s*\*\*Observações\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Avaliação de Risco": r'-\s*\*\*Avaliação de Risco\*\*:\s*(.*?)(?=-\s*\*\*|\Z)',
            "Grau de Confiança na Avaliação de Risco": r'-\s*\*\*Grau de Confiança na Avaliação de Risco\*\*:\s*(.*?)(?=\Z)'
        }
    }
    
    if format_type in patterns and section_name in patterns[format_type]:
        pattern = patterns[format_type][section_name]
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = match.group(1).strip()
            return result
    
    return "N/A"

def parse_event_folder(event_folder, format_type=None):
    """Parse an event folder and extract relevant information"""
    folder_name = os.path.basename(event_folder)
    image_count = count_images(event_folder)
    first_image_path, first_image_filename = get_first_image(event_folder)
    
    cam_id, event_datetime = parse_image_filename(first_image_filename)
    
    # Check for analysis.json first
    analysis_json_file = os.path.join(event_folder, "analysis.json")
    scenario_type = None
    
    if os.path.exists(analysis_json_file):
        try:
            with open(analysis_json_file, 'r', encoding='utf-8') as f:
                analysis_json = json.load(f)
                if "scenario_type" in analysis_json:
                    scenario_type = analysis_json["scenario_type"]
                    logger.info(f"Found scenario_type '{scenario_type}' in analysis.json for {folder_name}")
        except Exception as e:
            logger.warning(f"Error reading analysis.json in {folder_name}: {e}")
    
    # Fall back to classification function if needed
    if scenario_type is None:
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
            "label": "N/A",
            "avaliacao": "N/A",
            "scenario_type": scenario_type,
            "format_type": format_type or "default"
        }
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Auto-detect format if not specified
            detected_format = format_type or detect_format_type(content)
            
            if detected_format == "default":
                contexto = extract_section(content, "Contexto", "default")
                pessoas = extract_section(content, "Pessoas e veículos", "default")
                comportamentos = extract_section(content, "Comportamentos e sinais de risco", "default")
                label = extract_section(content, "Label", "default")
                avaliacao = extract_section(content, "Avaliação Geral de Risco", "default")
                
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
                    "scenario_type": scenario_type,
                    "format_type": "default"
                }
            else:  # detailed format
                contexto = extract_section(content, "Contexto", "detailed")
                observacoes = extract_section(content, "Observações", "detailed")
                avaliacao_risco = extract_section(content, "Avaliação de Risco", "detailed")
                grau_confianca = extract_section(content, "Grau de Confiança na Avaliação de Risco", "detailed")
                
                return {
                    "folder_name": folder_name,
                    "image_count": image_count,
                    "first_image": first_image_path,
                    "cam_id": cam_id,
                    "event_datetime": event_datetime,
                    "contexto": contexto,
                    "observacoes": observacoes,
                    "avaliacao": avaliacao_risco,
                    "grau_confianca": grau_confianca,
                    "scenario_type": scenario_type,
                    "format_type": "detailed"
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
            "avaliacao": "N/A",
            "scenario_type": scenario_type,
            "format_type": format_type or "default"
        }

def is_dangerous_camera(cam_id):
    """Determine if the camera ID is for a dangerous event"""
    dangerous_labels = [
        'hit', 'throw', 
        'struggle', 'kick', 
        'gun', 'videoplayback', 
        'altercation', 'burglary', 'climbing-wall',
        'break-in', 'car-break-in',  'robbery',
    ]
    
    if cam_id is None:
        return False
    
    return any(label.lower() in str(cam_id).lower() for label in dangerous_labels)

def is_predicted_dangerous(avaliacao, format_type="default"):
    """Determine if the event is predicted as dangerous based on avaliacao"""
    if avaliacao is None or avaliacao == "N/A":
        return False
    
    if format_type == "default":
        return "**Sim**" in avaliacao
    elif format_type == "detailed":
        return "Alto" in avaliacao.split('.')[0]  # Check before justification
    
    return False

def detect_format_type(analysis_content):
    """Detect the format type of the analysis content"""
    if re.search(r'-\s*\*\*Observações\*\*:', analysis_content):
        return "detailed"
    return "default"

def extract_confidence_level(grau_confianca):
    """Extract confidence level from the grau_confianca field"""
    if grau_confianca is None or grau_confianca == "N/A":
        return "Unknown"
    
    # Handle non-string types (like float or int)
    if not isinstance(grau_confianca, str):
        return "Unknown"
    
    if "Alto" in grau_confianca:
        return "Alto"
    elif "Médio" in grau_confianca or "Medio" in grau_confianca:
        return "Médio"
    elif "Baixo" in grau_confianca:
        return "Baixo"
    else:
        return "Unknown"

def generate_report(base_folder="motion_detected", output_file=None, format_type=None):
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
        data = parse_event_folder(folder, format_type)
        results.append(data)
    
    df = pd.DataFrame(results)
    
    df['real'] = df['cam_id'].apply(is_dangerous_camera)
    
    # Apply is_predicted_dangerous with the correct format type for each row
    df['predict'] = df.apply(lambda row: is_predicted_dangerous(row['avaliacao'], row.get('format_type', 'default')), axis=1)
    
    # Extract confidence level for detailed format
    if 'grau_confianca' in df.columns:
        # First make sure we handle missing values
        df['grau_confianca'] = df['grau_confianca'].fillna("N/A")
        df['confidence_level'] = df['grau_confianca'].apply(extract_confidence_level)
    
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
            # Overall metrics
            print("\n=== OVERALL METRICS ===")
            y_true = np.array(df['real'])
            y_pred = np.array(df['predict'])
            
            cm = confusion_matrix(y_true, y_pred)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print("\n=== Overall Confusion Matrix ===")
            print("              Predicted")
            print("              Not Dangerous  Dangerous")
            print(f"Actual Not Dangerous  {cm[0][0]:12d}  {cm[0][1]:9d}")
            print(f"      Dangerous       {cm[1][0]:12d}  {cm[1][1]:9d}")
            
            print("\n=== Overall Classification Metrics ===")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            
            # Metrics for each confidence level if we have detailed format
            if 'confidence_level' in df.columns and len(df[df['format_type'] == 'detailed']) > 0:
                print("\n\n=== METRICS BY CONFIDENCE LEVEL ===")
                confidence_levels = ['Alto', 'Médio', 'Baixo']
                
                for confidence in confidence_levels:
                    # Filter dataframe for current confidence level
                    conf_df = df[df['confidence_level'] == confidence]
                    
                    if len(conf_df) == 0:
                        print(f"\n--- No data found for confidence level: {confidence} ---")
                        continue
                    
                    print(f"\n=== CONFIDENCE LEVEL: {confidence.upper()} ({len(conf_df)} events) ===")
                    
                    # Calculate metrics for this confidence level
                    y_true_conf = np.array(conf_df['real'])
                    y_pred_conf = np.array(conf_df['predict'])
                    
                    # Skip if we don't have both classes
                    if len(set(y_true_conf)) < 2 or len(set(y_pred_conf)) < 2:
                        print(f"Warning: Not enough class variation for {confidence} confidence. Classes found: Real {set(y_true_conf)}, Predicted {set(y_pred_conf)}")
                        continue
                    
                    cm_conf = confusion_matrix(y_true_conf, y_pred_conf)
                    
                    accuracy_conf = accuracy_score(y_true_conf, y_pred_conf)
                    precision_conf = precision_score(y_true_conf, y_pred_conf, zero_division=0)
                    recall_conf = recall_score(y_true_conf, y_pred_conf, zero_division=0)
                    f1_conf = f1_score(y_true_conf, y_pred_conf, zero_division=0)
                    
                    print("\n=== Confusion Matrix ===")
                    print("              Predicted")
                    print("              Not Dangerous  Dangerous")
                    try:
                        print(f"Actual Not Dangerous  {cm_conf[0][0]:12d}  {cm_conf[0][1]:9d}")
                        print(f"      Dangerous       {cm_conf[1][0]:12d}  {cm_conf[1][1]:9d}")
                    except IndexError:
                        print("Incomplete confusion matrix due to missing classes")
                    
                    print("\n=== Classification Metrics ===")
                    print(f"Accuracy:  {accuracy_conf:.4f}")
                    print(f"Precision: {precision_conf:.4f}")
                    print(f"Recall:    {recall_conf:.4f}")
                    print(f"F1 Score:  {f1_conf:.4f}")
            
            # Metrics for each scenario type
            scenario_types = [
                'camera_analyzer_prompt_external_street',
                'camera_analyzer_prompt_internal_closed',
                'camera_analyzer_prompt_internal_to_outside'
            ]
            
            print("\n\n=== METRICS BY SCENARIO TYPE ===")
            for scenario in scenario_types:
                print(f"\n--- SCENARIO: {scenario} ---")
                
                # Filter dataframe for current scenario
                scenario_df = df[df['scenario_type'] == scenario]
                
                if len(scenario_df) == 0:
                    print(f"No data found for scenario type: {scenario}")
                    continue
                
                print(f"Number of events: {len(scenario_df)}")
                
                # Calculate metrics for this scenario
                y_true_scenario = np.array(scenario_df['real'])
                y_pred_scenario = np.array(scenario_df['predict'])
                
                # Skip if we don't have both classes
                if len(set(y_true_scenario)) < 2 or len(set(y_pred_scenario)) < 2:
                    print(f"Warning: Not enough class variation for {scenario}. Classes found: Real {set(y_true_scenario)}, Predicted {set(y_pred_scenario)}")
                    continue
                
                cm_scenario = confusion_matrix(y_true_scenario, y_pred_scenario)
                
                accuracy_scenario = accuracy_score(y_true_scenario, y_pred_scenario)
                precision_scenario = precision_score(y_true_scenario, y_pred_scenario, zero_division=0)
                recall_scenario = recall_score(y_true_scenario, y_pred_scenario, zero_division=0)
                f1_scenario = f1_score(y_true_scenario, y_pred_scenario, zero_division=0)
                
                print("\n=== Confusion Matrix ===")
                print("              Predicted")
                print("              Not Dangerous  Dangerous")
                try:
                    print(f"Actual Not Dangerous  {cm_scenario[0][0]:12d}  {cm_scenario[0][1]:9d}")
                    print(f"      Dangerous       {cm_scenario[1][0]:12d}  {cm_scenario[1][1]:9d}")
                except IndexError:
                    print("Incomplete confusion matrix due to missing classes")
                
                print("\n=== Classification Metrics ===")
                print(f"Accuracy:  {accuracy_scenario:.4f}")
                print(f"Precision: {precision_scenario:.4f}")
                print(f"Recall:    {recall_scenario:.4f}")
                print(f"F1 Score:  {f1_scenario:.4f}")
                
                # If we have detailed format, also do metrics by scenario AND confidence
                if 'confidence_level' in df.columns and len(scenario_df[scenario_df['format_type'] == 'detailed']) > 0:
                    confidence_levels = ['Alto', 'Médio', 'Baixo']
                    
                    for confidence in confidence_levels:
                        # Filter to get data for this scenario and confidence level
                        scen_conf_df = scenario_df[scenario_df['confidence_level'] == confidence]
                        
                        if len(scen_conf_df) == 0:
                            continue  # Skip if no data
                        
                        print(f"\n--- SCENARIO: {scenario}, CONFIDENCE: {confidence} ({len(scen_conf_df)} events) ---")
                        
                        y_true_sc = np.array(scen_conf_df['real'])
                        y_pred_sc = np.array(scen_conf_df['predict'])
                        
                        # Skip if we don't have both classes
                        if len(set(y_true_sc)) < 2 or len(set(y_pred_sc)) < 2:
                            print(f"Warning: Not enough class variation. Classes found: Real {set(y_true_sc)}, Predicted {set(y_pred_sc)}")
                            continue
                        
                        try:
                            cm_sc = confusion_matrix(y_true_sc, y_pred_sc)
                            
                            accuracy_sc = accuracy_score(y_true_sc, y_pred_sc)
                            precision_sc = precision_score(y_true_sc, y_pred_sc, zero_division=0)
                            recall_sc = recall_score(y_true_sc, y_pred_sc, zero_division=0)
                            f1_sc = f1_score(y_true_sc, y_pred_sc, zero_division=0)
                            
                            print("\n=== Confusion Matrix ===")
                            print("              Predicted")
                            print("              Not Dangerous  Dangerous")
                            print(f"Actual Not Dangerous  {cm_sc[0][0]:12d}  {cm_sc[0][1]:9d}")
                            print(f"      Dangerous       {cm_sc[1][0]:12d}  {cm_sc[1][1]:9d}")
                            
                            print("\n=== Classification Metrics ===")
                            print(f"Accuracy:  {accuracy_sc:.4f}")
                            print(f"Precision: {precision_sc:.4f}")
                            print(f"Recall:    {recall_sc:.4f}")
                            print(f"F1 Score:  {f1_sc:.4f}")
                        except Exception as e:
                            print(f"Could not calculate metrics: {e}")
                
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    print("\n=== Analysis Report Summary ===")
    print(f"Total events analyzed: {len(results)}")
    print("\nSample of the report (first 5 rows):")
    
    # Adjust columns to show based on available format types in the data
    display_columns = ["folder_name", "image_count", "avaliacao", "real", "predict", "scenario_type"]
    if "format_type" in df.columns:
        display_columns.append("format_type")
    if "grau_confianca" in df.columns:
        display_columns.append("grau_confianca")
    if "confidence_level" in df.columns:
        display_columns.append("confidence_level")
    
    print(df[display_columns].head().to_string())
    print(f"\nFull report saved to: {output_file}")
    
    # Breakdown by scenario type
    print("\nBreakdown by scenario type:")
    scenario_counts = df['scenario_type'].value_counts()
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} events")
    
    if "format_type" in df.columns:
        print("\nBreakdown by format type:")
        format_counts = df['format_type'].value_counts()
        for format_type, count in format_counts.items():
            print(f"  {format_type}: {count} events")
    
    # Breakdown by confidence level (if detailed format exists)
    if "confidence_level" in df.columns:
        print("\nBreakdown by confidence level:")
        confidence_counts = df['confidence_level'].value_counts()
        for confidence, count in confidence_counts.items():
            print(f"  {confidence}: {count} events")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate an analysis report from motion detection events')
    parser.add_argument('-i', '--input-folder', default='motion_detected',
                      help='Base folder containing event subfolders (default: motion_detected)')
    parser.add_argument('-o', '--output-file', 
                      help='Output file name (default: analysis_report_TIMESTAMP.csv)')
    parser.add_argument('-f', '--format', choices=['default', 'detailed'],
                      help='Format type for analysis files (default: auto-detect)')
    
    args = parser.parse_args()
    
    df = generate_report(base_folder=args.input_folder, output_file=args.output_file, format_type=args.format)