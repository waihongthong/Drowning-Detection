# api_server.py - Separate FastAPI server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
from PIL import Image
import io
from typing import Dict, Any
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="Drowning Detection API")

# Load model
try:
    model = YOLO('best2.pt')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class DetectionRequest(BaseModel):
    image: str
    confidence_threshold: float = 0.7

def detect_drowning(image_b64: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """Process image and return detections"""
    try:
        if model is None:
            return {
                'success': False,
                'error': 'Model not loaded',
                'detections': []
            }
        
        # Decode base64 image
        img_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_data))
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
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

@app.post("/detect")
async def detect_endpoint(request: DetectionRequest):
    """Detection endpoint"""
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image provided")
        
        result = detect_drowning(request.image, request.confidence_threshold)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_error",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cloud/status")
async def cloud_status_endpoint():
    """Cloud status endpoint"""
    return {
        "status": "healthy" if model is not None else "model_error",
        "service": "drowning_detection",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cloud/test")
async def cloud_test_endpoint():
    """Cloud test endpoint"""
    return {
        "message": "API is working",
        "service": "drowning_detection",
        "endpoints": ["/detect", "/health", "/api/cloud/status", "/api/cloud/test"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
