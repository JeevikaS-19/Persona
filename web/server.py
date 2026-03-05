import os
import sys
import json
import base64
import cv2
import numpy as np
import asyncio
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add parent directory to path to import specialist modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../py_file')))
from main import analyze_video_production

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('start_webcam_analysis')
def handle_webcam_start(data):
    print("Webcam session started")
    # This would initiate a live session handler
    # For now, we wait for 'process_frame' events

@socketio.on('process_frame')
def handle_frame(data):
    """Handles real-time frame processing from webcam."""
    try:
        # Decode base64 image
        header, encoded = data['image'].split(",", 1)
        data_decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(data_decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # In a real live scenario, we'd buffer frames and run rPPG
        # For the prototype, we signal back that we received it
        emit('frame_processed', {"status": "ok", "timestamp": data.get('timestamp')})
        
        # Real logic would emit 'rppg_update' here if we had enough buffer
    except Exception as e:
        print(f"Frame Error: {e}")

@socketio.on('upload_video')
def handle_upload(data):
    """Initiates batch analysis on an uploaded file."""
    # Note: In a real app, file would be streamed or saved to tmp
    # Here we assume the client provides a path for testing or we save it
    pass

async def run_analysis_wrapper(path):
    def signal_cb(value):
        socketio.emit('rppg_update', {"value": value})
        
    result = await analyze_video_production(path, signal_callback=signal_cb)
    socketio.emit('analysis_complete', result)

@socketio.on('trigger_analysis')
def trigger_analysis(data):
    """Triggers the existing main.py logic on a local file path."""
    path = data.get('path')
    if path and os.path.exists(path):
        asyncio.run(run_analysis_wrapper(path))
    else:
        emit('error', {"message": "Invalid path"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
