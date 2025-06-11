# app_local.py - Local testing version (FIXED - Updated for Gradio compatibility)
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

def api_detect(image_b64, confidence_threshold=0.7):
    """API-style function for programmatic access"""
    return detect_drowning(image_b64, confidence_threshold)

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
        
        ### Option 1: Use the separate FastAPI server (api_server.py)
        ```bash
        # Run the FastAPI server
        python api_server.py
        
        # Then make requests to http://localhost:8000
        ```
        
        ### Option 2: Use Gradio's built-in API
        ```python
        import requests
        import base64
        
        # First, convert your image to base64
        with open("your_image.jpg", "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode()
        
        # Use Gradio's API (when this app is running)
        response = requests.post(
            "http://localhost:5000/api/predict",
            json={
                "data": [img_b64, 0.7],  # [image_base64, confidence_threshold]
                "fn_index": 0  # Index of the gradio_detect function
            }
        )
        
        result = response.json()
        print(result)
        ```
        
        ### Option 3: Direct function call (if importing this module)
        ```python
        from app_local import api_detect
        
        # Your base64 image string
        result = api_detect(image_b64, confidence_threshold=0.7)
        print(result)
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
                "gradio_version": gr.__version__,
                "api_ready": True,
                "timestamp": datetime.now().isoformat()
            }
        
        health_btn.click(fn=health_check, outputs=health_output)
    
    with gr.Tab("Test API"):
        gr.Markdown("### Test the API functionality directly")
        
        with gr.Row():
            with gr.Column():
                test_image = gr.Image(type="pil", label="Test Image")
                test_confidence = gr.Slider(0.1, 1.0, 0.7, label="Confidence")
                test_btn = gr.Button("Test API Function")
            
            with gr.Column():
                test_output = gr.JSON(label="API Response")
        
        def test_api_function(image, confidence):
            if image is None:
                return {"error": "No image provided"}
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Test the API function
            return api_detect(img_b64, confidence)
        
        test_btn.click(
            fn=test_api_function,
            inputs=[test_image, test_confidence],
            outputs=test_output
        )

if __name__ == "__main__":
    # Launch on port 5000 for local testing
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
        show_api=True  # This enables Gradio's built-in API
    )
