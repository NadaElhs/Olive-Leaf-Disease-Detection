import sys
import io
import base64
import tempfile
import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/exp/weights/best.pt")

CLASS_COLORS = {
    0: "#22c55e",
    1: "#D7181B",
    2: "#271BAC",
}

app = FastAPI(
    title="Olive Disease Detection API",
    description="API de détection des maladies des oliviers via YOLOv8",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement du modèle ───────────────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"[✓] Modèle chargé : {MODEL_PATH}")
        for id, name in model.names.items():
            print(f"  {id} -> {name}")
    except Exception as e:
        print(f"[✗] Erreur : {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Schémas ────────────────────────────────────────────────────────────────────
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    color: str
    bbox: list

class PredictionResponse(BaseModel):
    success: bool
    detections: list
    total_detections: int
    image_annotated: str
    message: str

# ── Utilitaire : inférence sur un np array (Pour les Images) ───────────────────
def run_inference(img_np):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=img_bgr,
        imgsz=640,
        conf=0.40,
        iou=0.5,
        verbose=False,
    )
    result = results[0]
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = model.names.get(cls_id, f"Classe {cls_id}")
            color      = CLASS_COLORS.get(cls_id, "#6b7280")
            detections.append({
                "class_id":   cls_id,
                "class_name": class_name,
                "confidence": round(conf, 3),
                "color":      color,
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })

    # Annotation
    img_annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        hex_color = det["color"].lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (r, g, b), 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_annotated, (x1, y1-lh-10), (x1+lw+4, y1), (r, g, b), -1)
        cv2.putText(img_annotated, label, (x1+2, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    pil_img = Image.fromarray(img_annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return detections, img_b64

# ── Endpoint image (Classique) ─────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier doit être une image")

    try:
        contents = await file.read()
        image  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        detections, img_b64 = run_inference(img_np)

        sick = [d for d in detections if d["class_name"] != "healthy"]
        if not detections:
            msg = "Aucune détection — image non analysable"
        elif not sick:
            msg = "✅ Olivier sain — aucune maladie détectée"
        else:
            diseases = list({d["class_name"] for d in sick})
            msg = f"⚠️ Maladie(s) détectée(s) : {', '.join(diseases)}"

        return PredictionResponse(
            success=True,
            detections=detections,
            total_detections=len(detections),
            image_annotated=img_b64,
            message=msg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence : {str(e)}")

# ── NOUVEAU : Endpoint Vidéo Complète (Génère le .webm) ────────────────────────
@app.post("/predict-video-file")
async def predict_video_file_route(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # 1. Sauvegarder la vidéo envoyée par l'utilisateur
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    # 2. Remplacer l'extension par .webm pour la compatibilité web
    output_path = input_path.replace(suffix, "_annotated.mp4")

    try:
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Impossible de lire la vidéo")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 3. Préparer le nouveau fichier WEBM avec le codec vp09
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 4. Traitement frame par frame avec YOLO
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyse YOLO et annotation automatique
            results = model(frame, conf=0.15, iou=0.5, verbose=True)
            annotated_frame = results[0].plot()
            
            if annotated_frame.shape[:2] != (height, width):
                annotated_frame = cv2.resize(annotated_frame, (width, height))
            # Écrire dans la vidéo finale
            out.write(annotated_frame)

        cap.release()
        out.release()

    except Exception as e:
        if os.path.exists(output_path): os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")
    finally:
        if os.path.exists(input_path): os.remove(input_path)

    # 5. Renvoyer la vidéo terminée (en video/webm)
    # 5. Renvoyer la vidéo terminée
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"resultat_{file.filename}"
    )

# ── Autres endpoints ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Olive Disease Detection API", "status": "running", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.get("/classes")
def get_classes():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"classes": [{"id": k, "name": v, "color": CLASS_COLORS.get(k, "#6b7280")} for k, v in model.names.items()]}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)