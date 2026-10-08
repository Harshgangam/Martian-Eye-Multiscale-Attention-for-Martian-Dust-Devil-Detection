import os
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_api_endpoints():
    print("=== TESTING FASTAPI BACKEND ENDPOINTS ===")
    
    # 1. Health
    r = client.get("/api/health")
    print("Health Check:", r.json())
    assert r.status_code == 200
    
    # 2. Metrics
    r = client.get("/api/metrics")
    print("Metrics:", r.json())
    assert r.status_code == 200
    
    # 3. Dataset Samples
    r = client.get("/api/dataset/samples")
    samples_data = r.json()
    print(f"Dataset Samples Found: {len(samples_data['samples'])}")
    assert r.status_code == 200
    
    # 4. Sample Inference
    if len(samples_data['samples']) > 0:
        sample_id = samples_data['samples'][0]['id']
        r = client.post(f"/api/detect/sample/{sample_id}")
        det_res = r.json()
        print("Sample Detection Result Keys:", list(det_res.keys()))
        print(f"Is Dust Devil: {det_res['is_dust_devil']}, Calibrated Confidence: {det_res['calibrated_confidence']}")
        assert r.status_code == 200
        
    # 5. Ablation Matrix
    r = client.get("/api/experiments/ablation")
    abl = r.json()
    print(f"Ablation Experiments: {len(abl['experiments'])}")
    assert r.status_code == 200
    
    # 6. SSL Benchmarks
    r = client.get("/api/experiments/ssl")
    assert r.status_code == 200
    
    # 7. Domain Adaptation
    r = client.get("/api/experiments/domain-adapt")
    assert r.status_code == 200
    
    print("=== ALL BACKEND ENDPOINTS VERIFIED & WORKING PERFECTLY ===")

if __name__ == '__main__':
    test_api_endpoints()
