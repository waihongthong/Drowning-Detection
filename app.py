# app_local.py - Local testing version
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
from PIL import Image
import io
from typing import Dict, Any
from datetime import datetime

# Load your model (make sure best2.pt is in the same directory)
try:
    model = YOLO('best2.pt')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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

def gradio_detect(image, confidence_threshold=0.7):
    """Gradio interface function"""
    if image is None:
        return "❌ No image provided", None
    
    try:
        # Convert PIL to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Process detection
        result = detect_drowning(img_b64, confidence_threshold)
        
        # Create annotated image if detections found
        if result['success'] and result['detections']:
            img_array = np.array(image)
            
            for detection in result['detections']:
                x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
                conf = detection['confidence']
                class_name = detection['class']
                
                # Draw bounding box
                color = (255, 0, 0) if 'drown' in class_name.lower() else (0, 255, 0)
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img_array, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert back to PIL
            annotated_image = Image.fromarray(img_array)
        else:
            annotated_image = image
        
        # Format result text
        result_text = json.dumps(result, indent=2)
        
        return result_text, annotated_image
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Drowning Detection API - Local") as demo:
    gr.Markdown("# 🏊‍♂️ Drowning Detection System (Local Testing)")
    gr.Markdown("Upload an image to detect potential drowning incidents using AI")
    
    with gr.Tab("Web Interface"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                confidence_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.7, 
                    step=0.1, 
                    label="Detection Confidence Threshold"
                )
                detect_btn = gr.Button("🔍 Detect Drowning", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Detection Results (JSON)", 
                    lines=10,
                    show_copy_button=True
                )
                output_image = gr.Image(label="Annotated Image")
        
        detect_btn.click(
            fn=gradio_detect,
            inputs=[input_image, confidence_slider],
            outputs=[output_text, output_image]
        )
    
    with gr.Tab("API Documentation"):
        gr.Markdown("""
        ## 🔌 API Usage (Local Testing)
        
        **Endpoint**: `POST http://localhost:5000/detect`
        
        **Request Body**:
        ```json
        {
            "image": "base64_encoded_image_string",
            "confidence_threshold": 0.7
        }
        ```
        
        **Test Commands**:
        ```bash
        # Health check
        curl http://localhost:5000/health
        
        # Detection test (you'll need a base64 image)
        curl -X POST http://localhost:5000/detect \\
          -H "Content-Type: application/json" \\
          -d '{"image": "your_base64_image", "confidence_threshold": 0.7}'
        ```
        """)
    
    with gr.Tab("Health Check"):
        health_output = gr.JSON(label="System Status")
        health_btn = gr.Button("Check System Health")
        
        def health_check():
            return {
                "status": "healthy" if model is not None else "model_error",
                "model_loaded": model is not None,
                "model_classes": list(model.names.values()) if model else [],
                "api_ready": True,
                "timestamp": datetime.now().isoformat()
            }
        
        health_btn.click(fn=health_check, outputs=health_output)

# Custom API endpoints
@demo.fastapi_app.post("/detect")
async def direct_detect_endpoint(request: dict):
    """Direct API endpoint for /detect"""
    try:
        image_b64 = request.get("image", "")
        confidence_threshold = request.get("confidence_threshold", 0.7)
        
        if not image_b64:
            return {"success": False, "error": "No image provided", "detections": []}
        
        result = detect_drowning(image_b64, confidence_threshold)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e), "detections": []}

@demo.fastapi_app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_error",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@demo.fastapi_app.get("/api/cloud/status")
async def cloud_status_endpoint():
    """Cloud status endpoint (matching your curl test)"""
    return {
        "status": "healthy" if model is not None else "model_error",
        "service": "drowning_detection",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@demo.fastapi_app.get("/api/cloud/test")
async def cloud_test_endpoint():
    """Cloud test endpoint (matching your curl test)"""
    return {
        "message": "API is working",
        "service": "drowning_detection",
        "endpoints": ["/detect", "/health", "/api/cloud/status", "/api/cloud/test"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Launch on port 5000 for local testing
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
        show_api=True
    )
