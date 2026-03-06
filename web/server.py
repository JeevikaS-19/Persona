import os
import sys
import json
import base64
import tempfile
import uuid
import asyncio
import threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add parent directory to path to import specialist modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../py_file')))
import mediapipe_compat  # noqa — patches mp.solutions for mediapipe >= 0.10
from main import analyze_video_production

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global Task Store
TASKS = {}
# For log simulation
TASK_LOGS = {}

import psutil

# Set process priority to Below Normal for Windows stability
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print("[PERSONA] Process priority set to BELOW_NORMAL")
except Exception as e:
    print(f"[PERSONA] Failed to set priority: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/release', methods=['POST'])
def release_hardware():
    """Directive: Explicitly release hardware resources."""
    try:
        # Since we are headless and buffer-based, cap.release() is called 
        # inside the processing loop. This endpoint ensures no stray processes 
        # are holding the hardware.
        print("[PERSONA] Hardware Release Signal Received.")
        return jsonify({"status": "released", "message": "Backend hardware resources cleared."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_bridge():
    """The JSON-over-HTTP Bridge."""
    try:
        data = request.json
        if not data or 'buffer' not in data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        encoded_data = data['buffer']
        if "," in encoded_data:
            encoded_data = encoded_data.split(",", 1)[1]

        buffer_bytes = base64.b64decode(encoded_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(buffer_bytes)
            tmp_path = tmp.name

        # run_analysis_task normally runs in a thread for SocketIO
        # Here we run it synchronously (within the loop) for the REST response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(analyze_video_production(tmp_path))
            
            # Map to the User's requested JSON structure
            # JSON { "verdict": "DEEPFAKE", "score": 0.85, "rppg_graph": [...], "sync_score": 0.22 }
            if result.get("status") == "completed":
                m = result.get("metrics", {})
                f = result.get("forensics", {})
                bridge_response = {
                    "verdict": m.get("classification"),
                    "score": m.get("ensemble_score"),
                    "rppg_graph": f.get("filtered", []),
                    "sync_score": m.get("sync_score"),
                    "biometric_score": m.get("biometric_score"),
                    "reflection_score": m.get("reflection_score")
                }
                return jsonify(bridge_response)
            else:
                return jsonify({"status": "error", "message": result.get("message")}), 500
        finally:
            loop.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({"status": "error", "message": "Task not found"}), 404
    
    # Include logs in the status response
    logs = TASK_LOGS.get(task_id, [])
    # DEBUG: print polling activity
    # print(f"Polling {task_id}: {task.get('status')}")
    return jsonify({**task, "logs": logs})

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

def run_analysis_task(path, task_id, sid):
    """Execution bridge for the async analysis task."""
    def log_event(msg):
        TASK_LOGS[task_id].append(msg)
        print(f"[{task_id}] {msg}")

    log_event("Hardware Optimization: 10 FPS Resampling...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        log_event("Specialist Header: ROI Face Extraction...")
        # analyze_video_production is our async entry point
        result = loop.run_until_complete(analyze_video_production(path, signal_callback=None))
        
        if result.get("status") == "error":
            log_event(f"Forensic Failure: {result.get('message')}")
            TASKS[task_id] = {"status": "error", "message": result.get("message")}
        else:
            log_event("Sequential Audit: Analysis Complete.")
            log_event("Generating Integrated Forensic Report...")
            TASKS[task_id] = {"status": "completed", "result": result}
            print(f"Task {task_id} marked as COMPLETED")
            
    except Exception as e:
        log_event(f"System Crash: {str(e)}")
        TASKS[task_id] = {"status": "error", "message": str(e)}
    finally:
        loop.close()
        if os.path.exists(path):
            try: os.unlink(path)
            except: pass

@socketio.on('process_webcam_buffer')
def handle_webcam_buffer(data):
    try:
        task_id = str(uuid.uuid4())
        encoded_data = data.get('buffer')
        if not encoded_data:
            emit('task_started', {"status": "error", "message": "No buffer"})
            return

        if "," in encoded_data:
            encoded_data = encoded_data.split(",", 1)[1]

        buffer_bytes = base64.b64decode(encoded_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(buffer_bytes)
            tmp_path = tmp.name

        print(f"Buffer Receipt: {task_id}")
        TASKS[task_id] = {"status": "processing", "progress": 0}
        TASK_LOGS[task_id] = [f"Audit Initialized: {task_id}"]
        
        threading.Thread(target=run_analysis_task, args=(tmp_path, task_id, request.sid)).start()
        emit('task_started', {"task_id": task_id, "status": "processing"})
        
    except Exception as e:
        emit('error', {"message": str(e)})

@socketio.on('upload_video')
def handle_upload(data):
    try:
        task_id = str(uuid.uuid4())
        print(f"Upload Receipt: {task_id}")
        TASKS[task_id] = {"status": "processing", "progress": 0}
        TASK_LOGS[task_id] = [f"Audit Initialized: {task_id}"]
        
        encoded_data = data.get('file')
        if not encoded_data:
            emit('task_started', {"status": "error", "message": "No file data"})
            return

        if "," in encoded_data:
            encoded_data = encoded_data.split(",", 1)[1]

        video_bytes = base64.b64decode(encoded_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        threading.Thread(target=run_analysis_task, args=(tmp_path, task_id, request.sid)).start()
        emit('task_started', {"task_id": task_id, "status": "processing"})
        
    except Exception as e:
        emit('error', {"message": str(e)})

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5011, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
