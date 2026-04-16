import sys
import io
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn

# ── Configuration ──────────────────────────────────────────────────────────────
# Votre modèle est à la racine du projet (yolov8n.pt ou best.pt)
MODEL_PATH = Path("models/exp/weights/best.pt")  

CLASS_NAMES = {
    0: "healthy",
    1: "olive_peacock_spot",
    2: "aculus_olearius",
    
}

CLASS_COLORS = {
    0: "#22c55e",   # vert  – saine
    1: "#D7181B",   # rouge – aculus
    2: "#271BAC",   # bleu  – peacock spot
}

app = FastAPI(
    title="Olive Disease Detection API",
    description="API de détection des maladies des oliviers via YOLOv8",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement du modèle YOLOv8 ───────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"[✓] Modèle YOLOv8 chargé depuis : {MODEL_PATH}")
        print(f"[✓] Classes du modèle : {model.names}")
    except Exception as e:
        print(f"[✗] Erreur chargement modèle : {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()
    if model:
        print("Classes du modèle :", model.names)

# ── Schémas de réponse ─────────────────────────────────────────────────────────
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    color: str
    bbox: list  # [x1, y1, x2, y2]

class PredictionResponse(BaseModel):
    success: bool
    detections: list
    total_detections: int
    image_annotated: str
    message: str

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Olive Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier doit être une image")

    try:
        import cv2
        import numpy as np

        # Lecture de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        
    # ✅ Inférence YOLOv8 — simple et directe
        results = model.predict(
            source=img_np,
            imgsz=640,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )

        # ← DEBUG TEMPORAIRE
        result = results[0]
        print("=== DEBUG RAW ===")
        print("Boxes:", result.boxes)
        if result.boxes is not None:
            for box in result.boxes:
                print(f"  cls={int(box.cls[0])} conf={float(box.conf[0]):.4f} name={model.names[int(box.cls[0])]}")
        print("=================")

        detections = []
        result = results[0]  # première image

        # ✅ Extraction des boîtes YOLOv8
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Utilise les noms du modèle si CLASS_NAMES ne couvre pas l'ID
                class_name = CLASS_NAMES.get(cls_id, model.names.get(cls_id, f"Classe {cls_id}"))
                color      = CLASS_COLORS.get(cls_id, "#6b7280")

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=round(conf, 3),
                    color=color,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                ))

        # ✅ Annotation de l'image avec OpenCV
        img_annotated = img_np.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            hex_color = det.color.lstrip("#")
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            # Rectangle
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (r, g, b), 2)

            # Fond du label
            label = f"{det.class_name} {det.confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_annotated, (x1, y1 - lh - 10), (x1 + lw + 4, y1), (r, g, b), -1)

            # Texte blanc sur fond coloré
            cv2.putText(img_annotated, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encodage base64
        annotated_pil = Image.fromarray(img_annotated)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Message selon résultat
        sick_dets = [d for d in detections if d.class_name != "healthy"]
        if not detections:
            msg = "Aucune détection — image non analysable"
        elif not sick_dets:
            msg = "✅ Olivier sain — aucune maladie détectée"
        else:
            diseases = list({d.class_name for d in sick_dets})
            msg = f"⚠️ Maladie(s) détectée(s) : {', '.join(diseases)}"

        return PredictionResponse(
            success=True,
            detections=[d.dict() for d in detections],
            total_detections=len(detections),
            image_annotated=img_b64,
            message=msg,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence : {str(e)}")


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"id": k, "name": v, "color": CLASS_COLORS[k]}
            for k, v in CLASS_NAMES.items()
        ]
    }
@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    contents = await file.read()
    import numpy as np
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    # Test avec conf très bas pour forcer toutes les détections
    results = model.predict(source=img_np, imgsz=640, conf=0.01, iou=0.3, verbose=True)
    result = results[0]
    
    raw = []
    if result.boxes is not None:
        for box in result.boxes:
            raw.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
            })
    
    # Trier par confiance décroissante
    raw.sort(key=lambda x: x["confidence"], reverse=True)
    return {"total": len(raw), "detections": raw}

# ── Lancement ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

    