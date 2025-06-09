# app.py - Deploy this to Hugging Face Spaces
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
from PIL import Image
import io

# Load your model (upload your .pt file to the space)
model = YOLO('best2.pt')  # or whatever your model name is

def detect_drowning(image_b64, confidence_threshold=0.7):
    """Process image and return detections"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_data))
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(img_array, conf=confidence_threshold, verbose=False)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'confidence': float(conf),
                        'class': model.names[cls],
                        'class_id': cls
                    }
                    detections.append(detection)
        
        return {
            'success': True,
            'detections': detections,
            'total_detections': len(detections)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'detections': []
        }

# Create Gradio interface for API
def gradio_interface(image):
    """Gradio interface wrapper"""
    if image is None:
        return "No image provided"
    
    # Convert PIL to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Process
    result = detect_drowning(img_b64)
    return json.dumps(result, indent=2)

# Create the Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Drowning Detection API",
    description="Upload an image to detect drowning incidents"
)

# Also create API endpoints
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class DetectionRequest(BaseModel):
    image: str  # base64
    confidence_threshold: float = 0.7

@app.post("/detect")
async def api_detect(request: DetectionRequest):
    return detect_drowning(request.image, request.confidence_threshold)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

# Launch both Gradio and FastAPI
if __name__ == "__main__":
    # This will create both web interface and API
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Creates public URL
    )
