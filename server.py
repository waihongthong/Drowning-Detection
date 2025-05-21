from flask import Flask, Response, jsonify
import cv2
import threading
import json

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Use 0 for USB webcam or PiCamera

# Shared variable
detection_status = {"drowning_detected": False}

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(detection_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
