"""
FormCheck database layer
=========================
Extends the original single-user schema with:
  - users            (login accounts)
  - auth_tokens       (simple bearer tokens issued at login/register)
  - devices           (ESP32-CAM / ESP32 smartwatch registrations)
  - smartwatch_readings (HR / SpO2 / stress pushed by the custom ESP32 watch)

All of the original functions (log_session, log_challenge, get_user_sessions, ...)
are kept and now also power the multi-user flows used by server.py.
"""
import sqlite3
import secrets
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = 'user_data.db'


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # ── users / auth ──────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS auth_tokens (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # ── sessions / challenges (original schema, unchanged) ────
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        exercise TEXT NOT NULL,
        reps INTEGER NOT NULL,
        form_score REAL,
        heart_rate INTEGER,
        spo2 INTEGER,
        stress REAL,
        mood TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS challenge_logs (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        challenge_id TEXT NOT NULL,
        completed BOOLEAN NOT NULL,
        actual_reps INTEGER NOT NULL,
        form_score REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS model_meta (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        last_trained TIMESTAMP NOT NULL,
        completions_since_train INTEGER DEFAULT 0,
        model_accuracy REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # ── devices (ESP32-CAM / ESP32 smartwatch) ─────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        device_type TEXT NOT NULL,        -- 'esp32_cam' | 'esp32_watch'
        name TEXT NOT NULL,
        url TEXT,                          -- stream URL for esp32_cam (e.g. http://<ip>/stream)
        device_token TEXT UNIQUE NOT NULL, -- credential the physical device authenticates with
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS smartwatch_readings (
        id INTEGER PRIMARY KEY,
        device_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        heart_rate INTEGER,
        spo2 INTEGER,
        stress REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(device_id) REFERENCES devices(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    conn.commit()
    conn.close()
    print('✅ Database initialized (users, devices, sessions, challenges)')


# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════
TOKEN_TTL_DAYS = 30


def register_user(username, password, email=None):
    if not username or not password:
        return False, "Username and password required"
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
            (username, generate_password_hash(password), email)
        )
        conn.commit()
        user_id = c.lastrowid
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username already taken"
    finally:
        conn.close()


def _issue_token(user_id):
    token = secrets.token_hex(24)
    conn = get_db_connection()
    c = conn.cursor()
    expires = (datetime.utcnow() + timedelta(days=TOKEN_TTL_DAYS)).isoformat()
    c.execute('INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)',
              (token, user_id, expires))
    conn.commit()
    conn.close()
    return token


def login_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if not row or not check_password_hash(row['password_hash'], password):
        return False, "Invalid username or password"
    token = _issue_token(row['id'])
    return True, {'user_id': row['id'], 'username': username, 'token': token}


def get_user_from_token(token):
    """Resolve a bearer token to a user_id (or None if invalid/expired)."""
    if not token:
        return None
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''SELECT auth_tokens.user_id, auth_tokens.expires_at, users.username
                 FROM auth_tokens JOIN users ON users.id = auth_tokens.user_id
                 WHERE token = ?''', (token,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    if datetime.fromisoformat(row['expires_at']) < datetime.utcnow():
        return None
    return {'user_id': row['user_id'], 'username': row['username']}


# ═══════════════════════════════════════════════════════════════
# DEVICES (ESP32-CAM / ESP32 smartwatch)
# ═══════════════════════════════════════════════════════════════
def register_device(user_id, device_type, name, url=None):
    """Create (or replace) a device registration and return its device_token."""
    device_token = secrets.token_hex(16)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO devices (user_id, device_type, name, url, device_token)
                 VALUES (?, ?, ?, ?, ?)''', (user_id, device_type, name, url, device_token))
    conn.commit()
    device_id = c.lastrowid
    conn.close()
    return device_id, device_token


def list_devices(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM devices WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def delete_device(user_id, device_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM devices WHERE id = ? AND user_id = ?', (device_id, user_id))
    conn.commit()
    conn.close()


def get_device_by_token(device_token):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM devices WHERE device_token = ? AND is_active = 1', (device_token,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def touch_device(device_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE devices SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', (device_id,))
    conn.commit()
    conn.close()


def record_smartwatch_reading(device_id, user_id, heart_rate, spo2, stress):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO smartwatch_readings (device_id, user_id, heart_rate, spo2, stress)
                 VALUES (?, ?, ?, ?, ?)''', (device_id, user_id, heart_rate, spo2, stress))
    conn.commit()
    conn.close()


def get_latest_smartwatch_reading(user_id, max_age_seconds=30):
    """Return the most recent reading for the user's active smartwatch, or None if stale/missing."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''SELECT heart_rate, spo2, stress, timestamp FROM smartwatch_readings
                 WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1''', (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    age = (datetime.utcnow() - datetime.fromisoformat(row['timestamp'])).total_seconds()
    if age > max_age_seconds:
        return None
    return dict(row)


# ═══════════════════════════════════════════════════════════════
# SESSIONS / CHALLENGES (original functionality, per-user)
# ═══════════════════════════════════════════════════════════════
def log_session(user_id, exercise, reps, form_score, hr, spo2, stress, mood):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO sessions
        (user_id, exercise, reps, form_score, heart_rate, spo2, stress, mood)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, exercise, reps, form_score, hr, spo2, stress, mood))
    conn.commit()
    conn.close()


def log_challenge(user_id, challenge_id, completed, actual_reps, form_score):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO challenge_logs
        (user_id, challenge_id, completed, actual_reps, form_score)
        VALUES (?, ?, ?, ?, ?)''',
        (user_id, challenge_id, completed, actual_reps, form_score))
    conn.commit()
    conn.close()


def get_user_sessions(user_id, limit=50):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM sessions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
              (user_id, limit))
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results


def get_challenge_logs(user_id, limit=100):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM challenge_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
              (user_id, limit))
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results


def get_user_sessions_for_training(user_id, limit=500):
    """Sessions reshaped as the tuple format adaptive_model_kaggle.py expects:
    (id, user_id, exercise, reps, form_score, heart_rate, spo2, stress, mood, timestamp)
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''SELECT id, user_id, exercise, reps, form_score, heart_rate, spo2, stress, mood, timestamp
                 FROM sessions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return [tuple(r) for r in rows]


def get_unique_active_dates(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT DISTINCT substr(timestamp, 1, 10) as d FROM sessions WHERE user_id = ?', (user_id,))
    n = len(c.fetchall())
    conn.close()
    return n


if __name__ == '__main__':
    init_db()
