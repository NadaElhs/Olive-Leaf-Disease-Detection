import os
import io
import base64
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ── Configuration ──────────────────────────────────────────────────────────────
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 Mo

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


# ── API Proxy ──────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Proxy la requête vers FastAPI et retourne le résultat JSON."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier envoyé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Fichier vide"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Format non supporté. Utilisez PNG, JPG, JPEG ou WebP.",
        }), 400

    try:
        files = {"file": (file.filename, file.stream, file.content_type)}
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            files=files,
            timeout=60,
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            detail = response.json().get("detail", "Erreur inconnue")
            return jsonify({"success": False, "message": detail}), response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "message": "Impossible de joindre l'API FastAPI. Vérifiez qu'elle est lancée sur le port 8000.",
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({"success": False, "message": "Délai d'attente dépassé"}), 504
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


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


# ── Lancement ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Olive Disease Detection ")
    print("=" * 55)
    print(f"  Dashboard   : http://localhost:5000/dashboard")
    print(f"  Analyse     : http://localhost:5000/analyse")
    print(f"  Historique  : http://localhost:5000/historique")
    print(f"  FastAPI URL : {FASTAPI_URL}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=True)
