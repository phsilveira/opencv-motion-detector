import os
import logging
from openai import OpenAI
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenAIHelpers")

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 string for API transmission"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def classify_camera_scenario_with_api(image_path, api_key=None):
    """
    Classify a camera image into one of the predefined scenario types using OpenAI's Vision API
    
    Args:
        image_path: Path to the image file
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
        
    Returns:
        One of: camera_analyzer_prompt_external_street, camera_analyzer_prompt_internal_closed, 
        or camera_analyzer_prompt_internal_to_outside
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return "camera_analyzer_prompt_external_street"  # Default fallback
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Encode the image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "camera_analyzer_prompt_external_street"  # Default fallback
        
        # Classification prompt
        classification_prompt = """
        Classify this camera view into exactly one of the following three categories:
        
        1. external_street: Camera positioned externally, viewing a street, sidewalk, or outside area.
        2. internal_closed: Camera inside a closed environment without view to outside (rooms, halls, office).
        3. internal_to_outside: Camera inside looking toward outside areas like yards, gardens, or external corridors.
        
        Respond with ONLY the category name, nothing else.
        """
        
        # Make API request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=50
        )
        
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
        logger.error(f"Error classifying image with API: {e}")
        return "camera_analyzer_prompt_external_street"  # Default fallback

def analyze_event_with_api(images_folder, prompt_text, api_key=None):
    """
    Analyze a folder of event images using the specified prompt with OpenAI's Vision API
    
    Args:
        images_folder: Path to folder containing event images
        prompt_text: The system prompt text for analysis
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
        
    Returns:
        Analysis text result from GPT-4 Vision
    """
    client = OpenAI(api_key=api_key)
    
    try:
        # Get the first image (for demonstration)
        # In a real implementation, you might want to send multiple images or a video
        image_files = [f for f in os.listdir(images_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return "No images found in the event folder."
            
        # Sort images to ensure consistent order
        image_files.sort()
        
        # For demonstration, using just the first image
        # In practice, you would analyze multiple images or implement pagination
        image_path = os.path.join(images_folder, image_files[0])
        
        # Encode the image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Could not process the images."
        
        # Make API request
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analise esta imagem de segurança e forneça sua avaliação."},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        
        # Return the analysis
        return response.choices[0].message.content
            
    except Exception as e:
        logger.error(f"Error analyzing event with API: {e}")
        return f"Error analyzing event: {str(e)}"
