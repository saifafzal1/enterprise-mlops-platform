import io, os, subprocess, sys
import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms, models
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram

app = FastAPI(title="Enterprise MLOps API", version="2.0.0")
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


# ── Agentic SLM endpoints ─────────────────────────────────────────────────────

_slm_generation_total    = Counter("slm_generation_total",    "SLM script generation requests",  ["framework", "model"])
_slm_bmad_iterations     = Histogram("slm_bmad_iterations",   "BMAD correction iterations",      ["framework"])
_slm_validation_pass     = Counter("slm_validation_pass",     "BMAD validation passes",          ["framework"])

AGENTIC_MODELS_DIR = os.environ.get("AGENTIC_MODELS_DIR", "models/agentic")
BMAD_URL           = os.environ.get("BMAD_URL", "http://bmad-validator:8001")

_slm_models: dict = {}

def _get_slm(model_key: str):
    """Lazy-load fine-tuned SLM adapter."""
    if model_key in _slm_models:
        return _slm_models[model_key]
    model_dir = os.path.join(AGENTIC_MODELS_DIR, model_key)
    if not os.path.isdir(model_dir):
        return None
    _slm_models[model_key] = model_dir
    return model_dir


class GenerateRequest(BaseModel):
    story: str
    acceptance_criteria: Optional[List[str]] = []
    model: Optional[str] = "phi3"
    use_bmad: Optional[bool] = True
    max_bmad_iterations: Optional[int] = 3


@app.post("/generate-cypress")
def generate_cypress(req: GenerateRequest):
    model_dir = _get_slm(f"{req.model}_cypress")
    if model_dir is None:
        raise HTTPException(status_code=503,
            detail=f"Model {req.model}_cypress not loaded. Run fine-tuning first.")
    _slm_generation_total.labels(framework="cypress", model=req.model).inc()
    from src.agentic.bmad_agent import BMADAgent
    agent = BMADAgent(model_dir=model_dir, framework="cypress",
                      bmad_url=BMAD_URL, max_iterations=req.max_bmad_iterations)
    result = agent.run(req.story, req.acceptance_criteria) if req.use_bmad \
             else {"final_script": agent.generate(req.story, req.acceptance_criteria),
                   "valid": None, "iterations": 0}
    _slm_bmad_iterations.labels(framework="cypress").observe(result.get("iterations", 0))
    if result.get("valid"):
        _slm_validation_pass.labels(framework="cypress").inc()
    return result


@app.post("/generate-playwright")
def generate_playwright(req: GenerateRequest):
    model_dir = _get_slm(f"{req.model}_playwright")
    if model_dir is None:
        raise HTTPException(status_code=503,
            detail=f"Model {req.model}_playwright not loaded. Run fine-tuning first.")
    _slm_generation_total.labels(framework="playwright", model=req.model).inc()
    from src.agentic.bmad_agent import BMADAgent
    agent = BMADAgent(model_dir=model_dir, framework="playwright",
                      bmad_url=BMAD_URL, max_iterations=req.max_bmad_iterations)
    result = agent.run(req.story, req.acceptance_criteria) if req.use_bmad \
             else {"final_script": agent.generate(req.story, req.acceptance_criteria),
                   "valid": None, "iterations": 0}
    _slm_bmad_iterations.labels(framework="playwright").observe(result.get("iterations", 0))
    if result.get("valid"):
        _slm_validation_pass.labels(framework="playwright").inc()
    return result


@app.get("/agentic/models")
def list_agentic_models():
    available = []
    for key in ["phi3_cypress", "phi3_playwright", "gemma4_cypress", "gemma4_playwright"]:
        model_dir = os.path.join(AGENTIC_MODELS_DIR, key)
        available.append({"model": key, "ready": os.path.isdir(model_dir)})
    return {"models": available}
