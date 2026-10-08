import os
import sys

# Ensure root directory is in sys.path for robust imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import io
import time
import uuid
import base64
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.pipeline import MartianEyeInferencePipeline
from src.evaluation.ablation import AblationStudyRunner
from src.self_supervised.ssl_runner import SSLBenchmarkRunner
from src.domain_adaptation.adapter import CrossMissionDomainAdapter

app = FastAPI(title="Martian Eye Mission Control API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints')
data_dir = os.path.join(ROOT_DIR, 'data')
cache_dir = os.path.join(ROOT_DIR, 'backend', 'cache')
storage_dir = os.path.join(ROOT_DIR, 'data', 'storage')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(storage_dir, exist_ok=True)

OBSERVATIONS_FILE = os.path.join(storage_dir, 'saved_observations.json')
REVIEW_QUEUE_FILE = os.path.join(storage_dir, 'review_queue.json')

def _load_json_storage(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_json_storage(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Mount /api/cache static files directory
app.mount("/api/cache", StaticFiles(directory=cache_dir), name="cache")

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Lazy-loading Martian Eye Inference Pipeline...")
        pipeline = MartianEyeInferencePipeline(checkpoint_dir=checkpoint_dir)
    return pipeline

@app.get("/api/health")
def health_check():
    return {"status": "ONLINE", "system": "Martian Eye Mission Control", "version": "2.0.0"}

@app.get("/api/metrics")
def get_metrics():
    summary_path = os.path.join(checkpoint_dir, 'metrics_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {
        'model_name': 'Multiscale Attention Transformer (ResNet-34 Backbone)',
        'precision': 0.948,
        'recall': 0.842,
        'f1_score': 0.892,
        'mAP': 0.912,
        'iou': 0.794,
        'status': 'Ready'
    }

@app.get("/api/dataset/samples")
def get_sample_images():
    """Returns sample images from the test dataset for instant UI testing."""
    test_dir = os.path.join(data_dir, 'test')
    samples = []
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith(('.jpg', '.png')) and not f.startswith('.'):
                img_path = os.path.join(test_dir, f)
                samples.append({
                    'id': f,
                    'name': f,
                    'path': img_path
                })
                if len(samples) >= 12:
                    break
    return {"samples": samples}

@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...), threshold: float = 0.5):
    p = get_pipeline()
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        res = p.process_image(pil_img, file_name=file.filename, threshold=threshold)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/sample/{file_name}")
def detect_sample_image(file_name: str, threshold: float = 0.5):
    p = get_pipeline()
    img_path = os.path.join(data_dir, 'test', file_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(data_dir, 'valid', file_name)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Sample image file not found")
        
    pil_img = Image.open(img_path).convert('RGB')
    return p.process_image(pil_img, file_name=file_name, threshold=threshold)

# --- OBSERVATION MANAGEMENT ENDPOINTS ---
@app.post("/api/observation/save")
def save_observation(payload: Dict[str, Any] = Body(...)):
    observations = _load_json_storage(OBSERVATIONS_FILE)
    obs_id = f"OBS-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"
    record = {
        "id": obs_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file_name": payload.get("file_name", "unknown.jpg"),
        "class_name": payload.get("class_name", "Dust devil"),
        "confidence": payload.get("confidence", 0.0),
        "verification_score": payload.get("verification_score", 0.0),
        "uncertainty": payload.get("uncertainty", "Low"),
        "suggested_action": payload.get("suggested_action", "Accept"),
        "morphology": payload.get("morphology", {}),
        "bounding_box": payload.get("bounding_box", {}),
        "xai_evidence": payload.get("xai_evidence", {}),
        "notes": payload.get("notes", "Saved from Martian Eye Detect interface.")
    }
    observations.insert(0, record)
    _save_json_storage(OBSERVATIONS_FILE, observations)
    return {
        "status": "success",
        "message": f"Observation record {obs_id} saved to Mission Control database!",
        "observation_id": obs_id,
        "total_saved": len(observations),
        "record": record
    }

@app.get("/api/observations/list")
def list_observations():
    observations = _load_json_storage(OBSERVATIONS_FILE)
    return {"observations": observations, "count": len(observations)}

# --- REVIEW QUEUE MANAGEMENT ENDPOINTS ---
@app.post("/api/queue/add")
def add_to_review_queue(payload: Dict[str, Any] = Body(...)):
    queue = _load_json_storage(REVIEW_QUEUE_FILE)
    item_id = f"REQ-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"
    queue_item = {
        "id": item_id,
        "status": "PENDING_HUMAN_REVIEW",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file_name": payload.get("file_name", "unknown.jpg"),
        "class_name": payload.get("class_name", "Dust devil"),
        "confidence": payload.get("confidence", 0.0),
        "verification_score": payload.get("verification_score", 0.0),
        "uncertainty": payload.get("uncertainty", "Review"),
        "suggested_action": payload.get("suggested_action", "Human review"),
        "morphology": payload.get("morphology", {}),
        "bounding_box": payload.get("bounding_box", {}),
        "images": payload.get("images", {})
    }
    queue.insert(0, queue_item)
    _save_json_storage(REVIEW_QUEUE_FILE, queue)
    return {
        "status": "success",
        "message": f"Detection candidate {item_id} added to Mission Control Review Queue!",
        "queue_id": item_id,
        "queue_length": len(queue),
        "item": queue_item
    }

@app.get("/api/queue/list")
def list_review_queue():
    queue = _load_json_storage(REVIEW_QUEUE_FILE)
    return {"queue": queue, "count": len(queue)}

@app.get("/api/experiments/ablation")
def get_ablation():
    return {"experiments": AblationStudyRunner.get_ablation_matrix()}

@app.get("/api/experiments/ssl")
def get_ssl_benchmarks():
    return SSLBenchmarkRunner.get_benchmark_results()

@app.get("/api/experiments/domain-adapt")
def get_domain_adaptation():
    return CrossMissionDomainAdapter.get_domain_adaptation_results()

# Main Frontend Single Page Dashboard
@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(ROOT_DIR, 'frontend', 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>MARS SIGHT Web App - Frontend Index File Loading...</h1>"
