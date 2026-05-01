import os
import io
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

# ── Configuration ──────────────────────────────────────────────────────────────
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
UPLOAD_FOLDER = Path("uploads")

# MODIFIÉ : Ajout des formats vidéo (dont webm)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "mp4", "avi", "mov", "webm"}

# MODIFIÉ : Taille max augmentée à 100 Mo pour les vidéos
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "olive-disease-secret-key"

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def check_api_health() -> bool:
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("model_ready", False)
    except Exception:
        return False

# ── Routes principales ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analyse")
def analyse():
    return render_template("analyse.html")

@app.route("/historique")
def historique():
    return render_template("historique.html")

# ── API Proxy : IMAGE ──────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier envoyé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Fichier vide"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Format non supporté."}), 400

    try:
        files = {"file": (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{FASTAPI_URL}/predict", files=files, timeout=60)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"success": False, "message": "Erreur API"}), response.status_code
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ── API Proxy : VIDÉO ──────────────────────────────────────────────────────────

@app.route("/predict-video-file", methods=["POST"])
def predict_video_file():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier envoyé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Fichier vide"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Format vidéo non supporté."}), 400

    try:
        files = {"file": (file.filename, file.stream, file.content_type)}
        
        # Timeout très élevé (10 minutes) car l'IA traite chaque image de la vidéo
        print(f"Transfert de la vidéo vers l'IA...")
        response = requests.post(f"{FASTAPI_URL}/predict-video-file", files=files, timeout=600)

        if response.status_code == 200:
            from flask import Response as FlaskResponse
            # On renvoie la vidéo (en flux binaire WEBM) au navigateur
            return FlaskResponse(
                response.content,
                mimetype="video/mp4",
                headers={"Content-Disposition": "attachment; filename=resultat.mp4"}
            )
        else:
            detail = response.json().get("detail", "Erreur lors de l'analyse vidéo")
            return jsonify({"success": False, "message": detail}), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "message": "Le traitement de la vidéo prend trop de temps."}), 504
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ── Statut & Classes ───────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    ok = check_api_health()
    return jsonify({"api_ready": ok, "fastapi_url": FASTAPI_URL})

@app.route("/api/classes")
def classes():
    try:
        r = requests.get(f"{FASTAPI_URL}/classes", timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)