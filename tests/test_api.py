from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

HEART_PAYLOAD = {
    "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
}


def test_health_status_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_has_models_loaded_key():
    r = client.get("/health")
    assert "models_loaded" in r.json()


def test_predict_heart_returns_required_keys():
    r = client.post("/predict-heart", json=HEART_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "prediction"  in body
    assert "probability" in body
    assert "label"       in body


def test_predict_heart_prediction_is_binary():
    r = client.post("/predict-heart", json=HEART_PAYLOAD)
    assert r.json()["prediction"] in (0, 1)


def test_predict_heart_probability_range():
    r = client.post("/predict-heart", json=HEART_PAYLOAD)
    prob = r.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_heart_missing_field_returns_422():
    bad = {k: v for k, v in HEART_PAYLOAD.items() if k != "age"}
    r = client.post("/predict-heart", json=bad)
    assert r.status_code == 422


def test_predict_image_missing_file_returns_422():
    r = client.post("/predict-image")
    assert r.status_code == 422


def test_metrics_endpoint_exposed():
    r = client.get("/metrics")
    assert r.status_code == 200
