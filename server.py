from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from datetime import datetime
from threading import Thread

import database as db
from adaptive_model_kaggle import AdaptiveModelKaggle
from challenge_engine import get_daily_challenges

try:
    from challenge_engine import get_adaptive_daily_challenges
except ImportError:
    def get_adaptive_daily_challenges(model, form_score, heart_rate, spo2, stress):
        # Fallback: predict a difficulty label, then hand back that difficulty's static pool
        from challenge_engine import predict_difficulty
        difficulty = predict_difficulty(form_score, heart_rate, spo2, stress)
        return get_daily_challenges(difficulty)

app = Flask(__name__)
CORS(app)

db.init_db()
adaptive_model = AdaptiveModelKaggle()

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    print('⚠️ DeepFace unavailable')


# ═══════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════
def current_user():
    """Resolve the logged-in user from an Authorization: Bearer <token> header."""
    auth = request.headers.get('Authorization', '')
    token = auth.split(' ', 1)[1] if auth.lower().startswith('bearer ') else None
    if not token:
        token = request.args.get('token')
    return db.get_user_from_token(token)


def require_auth():
    user = current_user()
    if not user:
        return None, (jsonify({'error': 'unauthorized'}), 401)
    return user, None


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'formcheck-api',
        'model_status': adaptive_model.get_status(),
        'deepface': 'available' if DEEPFACE_AVAILABLE else 'fallback'
    }), 200


# ═══════════════════════════════════════════════════════════════
# AUTH API (used by streamlit_app.py, and by the HTML if run standalone)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.json or {}
    ok, result = db.register_user(data.get('username', '').strip(), data.get('password', ''), data.get('email'))
    if not ok:
        return jsonify({'error': result}), 400
    ok2, login_result = db.login_user(data.get('username').strip(), data.get('password'))
    return jsonify({'status': 'ok', **login_result}), 201


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    ok, result = db.login_user(data.get('username', '').strip(), data.get('password', ''))
    if not ok:
        return jsonify({'error': result}), 401
    return jsonify({'status': 'ok', **result}), 200


# ═══════════════════════════════════════════════════════════════
# DEVICES API (ESP32-CAM + custom ESP32 smartwatch)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/devices/register', methods=['POST'])
def register_device():
    user, err = require_auth()
    if err:
        return err
    data = request.json or {}
    device_type = data.get('device_type')
    name = data.get('name', device_type)
    url = data.get('url')
    if device_type not in ('esp32_cam', 'esp32_watch'):
        return jsonify({'error': "device_type must be 'esp32_cam' or 'esp32_watch'"}), 400
    device_id, device_token = db.register_device(user['user_id'], device_type, name, url)
    return jsonify({'status': 'ok', 'device_id': device_id, 'device_token': device_token}), 201


@app.route('/api/devices/list', methods=['GET'])
def list_devices():
    user, err = require_auth()
    if err:
        return err
    return jsonify({'devices': db.list_devices(user['user_id'])}), 200


@app.route('/api/devices/<int:device_id>', methods=['DELETE'])
def remove_device(device_id):
    user, err = require_auth()
    if err:
        return err
    db.delete_device(user['user_id'], device_id)
    return jsonify({'status': 'deleted'}), 200


@app.route('/api/devices/smartwatch/data', methods=['POST'])
def smartwatch_push():
    """The physical ESP32 smartwatch POSTs readings here using its own device_token
    (not a user login token — the watch never sees the user's password)."""
    data = request.json or {}
    device = db.get_device_by_token(data.get('device_token', ''))
    if not device or device['device_type'] != 'esp32_watch':
        return jsonify({'error': 'invalid device_token'}), 401
    db.record_smartwatch_reading(
        device['id'], device['user_id'],
        data.get('heart_rate'), data.get('spo2'), data.get('stress')
    )
    db.touch_device(device['id'])
    return jsonify({'status': 'logged'}), 200


@app.route('/api/devices/smartwatch/latest', methods=['GET'])
def smartwatch_latest():
    """Polled every few seconds by the browser dashboard to display live vitals."""
    user, err = require_auth()
    if err:
        return err
    reading = db.get_latest_smartwatch_reading(user['user_id'])
    if not reading:
        return jsonify({'available': False}), 200
    return jsonify({
        'available': True,
        'heart_rate': reading['heart_rate'],
        'spo2': reading['spo2'],
        'stress': reading['stress'],
        'timestamp': reading['timestamp'],
    }), 200


@app.route('/api/devices/esp32cam/active', methods=['GET'])
def esp32cam_active():
    """Returns the stream URL of the user's registered ESP32-CAM, if any."""
    user, err = require_auth()
    if err:
        return err
    devices = [d for d in db.list_devices(user['user_id']) if d['device_type'] == 'esp32_cam' and d['is_active']]
    if not devices:
        return jsonify({'available': False}), 200
    return jsonify({'available': True, 'url': devices[0]['url'], 'name': devices[0]['name']}), 200


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE CHALLENGES API
# ═══════════════════════════════════════════════════════════════
@app.route('/api/challenges/daily', methods=['GET'])
def get_daily():
    user, err = require_auth()
    if err:
        return err
    try:
        form_score = float(request.args.get('form_score', 80))
        heart_rate = int(request.args.get('heart_rate', 70))
        spo2 = int(request.args.get('spo2', 95))
        stress = float(request.args.get('stress', 50))

        user_sessions = db.get_user_sessions_for_training(user['user_id'])
        if len(user_sessions) > 5 and adaptive_model.metadata['foundation_trained']:
            daily = get_adaptive_daily_challenges(adaptive_model, form_score, heart_rate, spo2, stress)
        else:
            daily = get_daily_challenges('medium')

        return jsonify(daily), 200
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/challenges/log', methods=['POST'])
def log_challenge_completion():
    user, err = require_auth()
    if err:
        return err
    try:
        data = request.json or {}
        db.log_challenge(
            user['user_id'],
            data.get('challenge_id'),
            data.get('completed', False),
            data.get('actual_reps', 0),
            data.get('form_score', 0),
        )
        return jsonify({'status': 'logged'}), 200
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/challenges/retrain', methods=['POST'])
def retrain_model():
    user, err = require_auth()
    if err:
        return err
    try:
        user_sessions = db.get_user_sessions_for_training(user['user_id'], limit=500)
        if len(user_sessions) < 5:
            return jsonify({'status': 'insufficient_data', 'count': len(user_sessions), 'needed': 5}), 200

        success = adaptive_model.train(user_sessions)
        if success:
            return jsonify({'status': 'success', 'samples': len(user_sessions),
                             'model_status': adaptive_model.get_status()}), 200
        return jsonify({'status': 'training_failed'}), 400
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def model_status():
    user, err = require_auth()
    if err:
        return err
    try:
        user_sessions = db.get_user_sessions_for_training(user['user_id'])
        active_days = db.get_unique_active_dates(user['user_id'])

        status = adaptive_model.get_status()
        status['active_days'] = active_days
        status['total_sessions'] = len(user_sessions)
        status['days_to_user_data'] = max(0, 10 - active_days)
        status['ready_for_mixed_training'] = active_days >= 10
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    user, err = require_auth()
    if err:
        return err
    try:
        logs = db.get_challenge_logs(user['user_id'], limit=30)
        user_sessions = db.get_user_sessions_for_training(user['user_id'], limit=100)

        if not logs:
            return jsonify({'total_challenges': 0, 'avg_form': 0}), 200

        form_scores = [log['form_score'] for log in logs if log.get('form_score')]
        avg_form = float(np.mean(form_scores)) if form_scores else 0

        return jsonify({
            'total_challenges': len(logs),
            'avg_form': round(avg_form, 1),
            'best_form': max(form_scores) if form_scores else 0,
            'total_sessions': len(user_sessions),
            'model_confidence': adaptive_model.metadata['user_data_confidence'],
            'dataset': 'kaggle_fitness_tracker'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/log', methods=['POST'])
def api_log_session():
    """Log a completed workout session (reps/form/vitals) for the logged-in user."""
    user, err = require_auth()
    if err:
        return err
    data = request.json or {}
    db.log_session(
        user['user_id'], data.get('exercise', 'unknown'), data.get('reps', 0),
        data.get('form_score'), data.get('heart_rate'), data.get('spo2'),
        data.get('stress'), data.get('mood')
    )
    return jsonify({'status': 'logged'}), 200


# ═══════════════════════════════════════════════════════════════
# DEEPFACE ENDPOINT
# ═══════════════════════════════════════════════════════════════
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_emotion():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json or {}
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({'error': 'No image'}), 400

        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'mood': 'Neutral', 'confidence': 50, 'stress': 50}), 200

        if DEEPFACE_AVAILABLE:
            try:
                analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
                if analysis:
                    emotions = analysis[0]['emotion']
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = int(emotions[dominant_emotion])
                    stress = int((emotions.get('angry', 0) + emotions.get('fear', 0)) / 2)
                    return jsonify({'mood': dominant_emotion.capitalize(), 'confidence': confidence,
                                     'stress': stress}), 200
            except Exception:
                pass

        return jsonify({'mood': 'Neutral', 'confidence': 50, 'stress': 45}), 200
    except Exception:
        return jsonify({'mood': 'Neutral', 'confidence': 50, 'stress': 50}), 200


@app.route('/health-df', methods=['GET'])
def health_df():
    """Alternate health endpoint if DeepFace is embedded in this same server
    instead of running as a separate sidecar (see deepface_server.py)."""
    return jsonify({'status': 'ok', 'deepface': DEEPFACE_AVAILABLE}), 200


if __name__ == '__main__':
    print('''
    ╔══════════════════════════════════════════════╗
    ║   FormCheck Backend (multi-user, Kaggle ML)   ║
    ║   - API: :5000                                ║
    ║   - DeepFace sidecar: run deepface_server.py  ║
    ║     separately on :5001 (optional)            ║
    ╚══════════════════════════════════════════════╝
    ''')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
