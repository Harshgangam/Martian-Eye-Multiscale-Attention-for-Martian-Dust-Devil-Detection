import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_ui_and_detection():
    print("=== TESTING MARS SIGHT UI & DETECTION ===")
    
    # 1. Test index page
    r = client.get("/")
    assert r.status_code == 200
    print("Index page served successfully (HTML length:", len(r.text), ")")
    
    # 2. Test samples
    r = client.get("/api/dataset/samples")
    samples = r.json()['samples']
    print("Samples retrieved:", len(samples))
    assert len(samples) > 0
    
    # 3. Test detection output for first sample
    sample_id = samples[0]['id']
    r = client.post(f"/api/detect/sample/{sample_id}")
    assert r.status_code == 200
    res = r.json()
    
    print("\nDetection Telemetry:")
    print("  File:", res['file_name'])
    print("  Class:", res['class_name'])
    print("  Confidence:", res['confidence'])
    print("  Uncertainty:", res['uncertainty'])
    print("  Verification Score:", res['verification_score'])
    print("  Suggested Action:", res['suggested_action'])
    print("  Morphology:", res['morphology'])
    print("  Images Available:", list(res['images'].keys()))
    
    assert 'original' in res['images']
    assert 'detection' in res['images']
    assert 'attention' in res['images']
    assert 'contrast' in res['images']
    assert 'crop' in res['images']
    
    print("\n=== UI & MODEL DETECTION TEST PASSED PERFECTLY ===")

if __name__ == '__main__':
    test_ui_and_detection()
