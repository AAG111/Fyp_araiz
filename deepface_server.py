"""
DeepFace Sidecar Server & SQLite Backend
========================================
Tiny Flask server that sits alongside fitness_standalone.html.
Runs DeepFace.analyze() on face crops and manages a persistent SQLite 
database to track user exercise history.

INSTALL (one-time, run this in your terminal):
    pip install flask flask-cors deepface opencv-python numpy flask-sqlalchemy

RUN:
    python deepface_server.py

NOTE: On the very first run DeepFace downloads model weights
(~300 MB). Keep the terminal open until you see:
    DeepFace sidecar ready on http://localhost:5001
"""

import base64
import logging
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("deepface-sidecar")

app = Flask(__name__)
CORS(app)  # allow requests from file:// and localhost

# --- DATABASE SETUP ---
# Creates a local file named 'user_history.db' in the same folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    exercise = db.Column(db.String(20))
    reps = db.Column(db.Integer)
    form_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Ensure database tables are created before taking requests
with app.app_context():
    db.create_all()

# --- DEEPFACE SETUP ---
log.info("Loading DeepFace — first run will download ~300 MB of model weights...")
from deepface import DeepFace
log.info("DeepFace sidecar ready on http://localhost:5001")

# Map DeepFace raw labels -> human-friendly mood names shown on dashboard
MOOD_MAP = {
    "happy":    "Happy",
    "sad":      "Sad",
    "angry":    "Angry",
    "fear":     "Stressed",
    "disgust":  "Stressed",
    "surprise": "Energised",
    "neutral":  "Neutral",
}

# --- ROUTES ---

@app.route("/health", methods=["GET"])
def health():
    """Ping endpoint — HTML polls this to detect if sidecar is running."""
    return jsonify({"status": "ok", "model": "deepface", "database": "connected"})


@app.route('/api/log-session', methods=['POST'])
def log_session():
    """Logs a completed exercise set to the SQLite database."""
    try:
        data = request.get_json(force=True)
        session = UserSession(
            user_id='default',
            exercise=data.get('exercise', 'unknown'),
            reps=data.get('reps', 0),
            form_score=data.get('form_score', 0.0)
        )
        db.session.add(session)
        db.session.commit()
        return jsonify({'status': 'logged'})
    except Exception as e:
        log.error(f"Failed to log session: {e}")
        return jsonify({"error": "could not log session"}), 500


@app.route('/api/user-stats', methods=['GET'])
def get_stats():
    """Retrieves longitudinal progress for adaptive difficulty scaling."""
    sessions = UserSession.query.filter_by(user_id='default').all()
    
    if not sessions:
        return jsonify({'avg_form': 0, 'total_sessions': 0})
        
    avg_form = sum(s.form_score for s in sessions) / len(sessions)
    return jsonify({
        'avg_form': round(avg_form, 2), 
        'total_sessions': len(sessions)
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Request body (JSON):
        { "image": "<base64-encoded JPEG string>" }

    Response (JSON):
        {
            "mood":       "Happy",
            "confidence": 82,
            "stress":     14,
            "emotions":   { "happy": 82.1, "neutral": 10.3, ... }
        }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "missing 'image' field"}), 400

        # Decode base64 string -> OpenCV BGR image
        img_bytes = base64.b64decode(data["image"])
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "could not decode image"}), 400

        # Run DeepFace emotion analysis
        # enforce_detection=False: don't raise if face isn't perfectly centred
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        if not results:
            return jsonify({
                "mood": "Neutral", "confidence": 0,
                "stress": 0, "emotions": {}
            })

        r          = results[0]
        emotions   = r.get("emotion", {})
        dominant   = r.get("dominant_emotion", "neutral")
        mood       = MOOD_MAP.get(dominant, "Neutral")
        confidence = round(emotions.get(dominant, 0))

        # Stress = weighted blend of fear + angry + disgust, capped at 100
        stress = min(100, round(
            emotions.get("fear",    0) * 0.5 +
            emotions.get("angry",   0) * 0.4 +
            emotions.get("disgust", 0) * 0.3
        ))

        rounded_emotions = {k: round(v, 1) for k, v in emotions.items()}

        log.info("mood=%-10s  conf=%d%%  stress=%d%%" % (mood, confidence, stress))

        return jsonify({
            "mood":       mood,
            "confidence": confidence,
            "stress":     stress,
            "emotions":   rounded_emotions,
        })

    except Exception:
        log.error(traceback.format_exc())
        return jsonify({"error": "analysis failed"}), 500


if __name__ == "__main__":
    # threaded=False avoids TensorFlow/Keras threading issues
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)