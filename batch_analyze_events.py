#!/usr/bin/env python
# filepath: /Users/paulohenrique/Documents/freela/auria/motion-detection/opencv-motion-detector/batch_analyze_events.py
import os
import json
import base64
import logging
import asyncio
import argparse
import glob
from typing import List, Optional
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
        self.model = config.get("model", "chatgpt-4o-latest")
        self.force_reanalyze = config.get("force_reanalyze", False)
        self.start_from = config.get("start_from", None)
        self.dry_run = config.get("dry_run", False)
        self.filter = config.get("filter", None)
        
        # Initialize OpenAI client
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it in .env file or pass via --api-key")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Set up asyncio event loop
        self.loop = asyncio.get_event_loop()

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
        """Analyze a single event."""
        # Limit the number of images to analyze
        images_to_analyze = image_files[:self.max_images_to_analyze]
        
        # Analyze the images
        analysis = await self.analyze_snapshots(event_id, images_to_analyze)
        
        # Save the analysis
        self.save_analysis(event_id, event_folder, analysis)
    
    async def analyze_snapshots(self, event_id: str, snapshot_files: List[str]) -> str:
        """Analyze snapshots using OpenAI's vision capabilities."""
        base64_images = []
        for snapshot in snapshot_files:
            try:
                with open(snapshot, "rb") as f:
                    base64_images.append(base64.b64encode(f.read()).decode('utf-8'))
            except Exception as e:
                logger.error(f"Error reading image {os.path.basename(snapshot)}: {e}")
        
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
    - **Contexto**: Externo, noite, estacionamento.
    - **Pessoas e veículos**: indivíduo aproximando-se lentamente, tenta abrir porta ao final.
    - **Comportamentos e sinais de risco**: tentativa clara de acesso indevido, pedaços possivelmente da porta quebrados no chão.
    - **Label**: Estacionamento 4.
    - **Avaliação Geral de Risco**: **Sim**, tentativa explícita de invasão."""},
            {"role": "user", "content": content_list}
        ]
        
        try:
            logger.info(f"Sending request to OpenAI API for event {event_id}")
            params = {
                "model": self.model,
                "messages": prompt_messages,
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            response = self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API for event {event_id}: {e}")
            return f"Failed to analyze the snapshots: {str(e)}"
    
    def save_analysis(self, event_id: str, event_folder: str, analysis: str):
        """Save the analysis results for an event."""
        if not os.path.exists(event_folder):
            logger.warning(f"Event directory {event_folder} does not exist. Analysis not saved.")
            return
        
        result_data = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "api_version": "gpt-4o"
        }
        
        try:
            # Save text version
            with open(os.path.join(event_folder, "analysis.txt"), 'w') as f:
                f.write(analysis)
            
            # Save JSON version
            with open(os.path.join(event_folder, "analysis.json"), 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Analysis saved for event {event_id}")
        except Exception as e:
            logger.error(f"Error saving analysis for event {event_id}: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch analyze motion detection events')
    
    parser.add_argument('-i', '--input-folder', default='motion_detected',
                        help='Base folder containing event subfolders (default: motion_detected)')
    parser.add_argument('--api-key', 
                        help='OpenAI API key (default: read from OPENAI_API_KEY environment variable)')
    parser.add_argument('--max-images', type=int, default=20,
                        help='Maximum number of images to analyze per event (default: 100)')
    parser.add_argument('--model', default='chatgpt-4o-latest',
                        help='OpenAI model to use (default: chatgpt-4o-latest)')
    parser.add_argument('--force', action='store_true',
                        help='Force reanalysis of events even if they have existing analysis files')
    parser.add_argument('--start-from', 
                        help='Start analysis from a specific event ID (partial match)')
    parser.add_argument('--filter', 
                        help='Only analyze events with this string in their ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be analyzed without making API calls')
    
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