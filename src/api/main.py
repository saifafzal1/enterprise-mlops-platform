import io, os
import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms, models
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

app = FastAPI(title="Enterprise MLOps API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

# Required by the FastAPI Observability dashboard variable query
_app_info = Gauge("fastapi_app_info", "FastAPI application info", ["app_name"])
_app_info.labels(app_name="mlops-api").set(1)

LABELS = ["cat", "dog"]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_image_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load("models/cats_vs_dogs.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m


MODELS_LOADED = False
img_model     = None
heart_bundle  = None

try:
    img_model    = _load_image_model()
    heart_bundle = joblib.load("models/heart_disease.pkl")
    MODELS_LOADED = True
except Exception as e:
    print(f"[warn] Models not loaded at startup — {e}")


class HeartFeatures(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if img_model is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Image model not loaded")
    img    = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = IMG_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(img_model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return {"label": LABELS[idx], "confidence": round(probs[idx].item(), 4)}


@app.post("/predict-heart")
def predict_heart(features: HeartFeatures):
    if heart_bundle is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Heart model not loaded")
    X    = np.array([[features.age, features.sex, features.cp, features.trestbps,
                      features.chol, features.fbs, features.restecg, features.thalach,
                      features.exang, features.oldpeak, features.slope, features.ca, features.thal]])
    X_sc = heart_bundle["scaler"].transform(X)
    pred = heart_bundle["model"].predict(X_sc)[0]
    prob = heart_bundle["model"].predict_proba(X_sc)[0][1]
    return {
        "prediction":  int(pred),
        "probability": round(float(prob), 4),
        "label":       "disease" if pred == 1 else "no disease",
    }
