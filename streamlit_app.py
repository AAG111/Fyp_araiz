"""
FormCheck — Streamlit host
===========================
This file does NOT redesign the dashboard. It:
  1. Gates access behind a login/register screen (talks to server.py's /api/auth/*).
  2. Offers a small sidebar panel to connect an ESP32-CAM and/or a custom ESP32
     smartwatch (talks to server.py's /api/devices/*).
  3. Embeds the original fitness_gamified.html untouched, injecting the user's
     token / API URLs / device URLs as JS globals it already knows how to read.

Run:
    python server.py                 # terminal 1 — Flask API on :5000
    python deepface_server.py        # terminal 2 — optional, DeepFace sidecar on :5001
    streamlit run streamlit_app.py   # terminal 3 — this file, on :8501
"""
import streamlit as st
import requests
from pathlib import Path

st.set_page_config(page_title="FormCheck — Gamified AI Fitness", layout="wide", initial_sidebar_state="expanded")

HTML_PATH = Path(__file__).parent / "fitness_gamified.html"

# ── configurable backend location (defaults work for same-machine dev) ──
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:5000"
if 'df_url' not in st.session_state:
    st.session_state.df_url = "http://localhost:5001"
for key, default in [('logged_in', False), ('user_id', None), ('username', None), ('token', None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def api(path, method="GET", auth=True, **kwargs):
    url = st.session_state.api_url.rstrip('/') + path
    headers = kwargs.pop('headers', {})
    if auth and st.session_state.token:
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    try:
        r = requests.request(method, url, headers=headers, timeout=6, **kwargs)
        return r
    except requests.RequestException as e:
        return None


# ═══════════════════════════════════════════════════════════════
# LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════
def show_login():
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center;margin-top:40px;'>
            <h1 style='font-size:44px;margin:0;color:#00ff87;'>⚡ FormCheck</h1>
            <p style='color:#ffcc00;margin-top:6px;'>Gamified AI Fitness Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        with st.expander("⚙️ Backend server address", expanded=False):
            st.session_state.api_url = st.text_input("API server URL", st.session_state.api_url)
            st.session_state.df_url = st.text_input("DeepFace sidecar URL (optional)", st.session_state.df_url)
            st.caption("Only change these if server.py / deepface_server.py run on a different machine or port.")

        tab_login, tab_register = st.tabs(["LOGIN", "REGISTER"])

        with tab_login:
            username = st.text_input("Username", key="login_u")
            password = st.text_input("Password", type="password", key="login_p")
            if st.button("⚡ LOGIN", type="primary", use_container_width=True):
                r = api("/api/auth/login", "POST", auth=False, json={"username": username, "password": password})
                if r is None:
                    st.error(f"Can't reach the API server at {st.session_state.api_url} — is server.py running?")
                elif r.status_code == 200:
                    d = r.json()
                    st.session_state.logged_in = True
                    st.session_state.user_id = d['user_id']
                    st.session_state.username = d['username']
                    st.session_state.token = d['token']
                    st.rerun()
                else:
                    st.error(r.json().get('error', 'Login failed'))

        with tab_register:
            new_u = st.text_input("Choose username", key="reg_u")
            new_p = st.text_input("Choose password", type="password", key="reg_p")
            confirm_p = st.text_input("Confirm password", type="password", key="reg_c")
            email = st.text_input("Email (optional)", key="reg_e")
            if st.button("📝 REGISTER", type="primary", use_container_width=True):
                if not new_u or not new_p:
                    st.error("Username and password required")
                elif new_p != confirm_p:
                    st.error("Passwords don't match")
                elif len(new_p) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    r = api("/api/auth/register", "POST", auth=False,
                            json={"username": new_u, "password": new_p, "email": email})
                    if r is None:
                        st.error(f"Can't reach the API server at {st.session_state.api_url} — is server.py running?")
                    elif r.status_code == 201:
                        d = r.json()
                        st.session_state.logged_in = True
                        st.session_state.user_id = d['user_id']
                        st.session_state.username = d['username']
                        st.session_state.token = d['token']
                        st.success("Account created — welcome!")
                        st.rerun()
                    else:
                        st.error(r.json().get('error', 'Registration failed'))


# ═══════════════════════════════════════════════════════════════
# DEVICE LINKING (sidebar) — ESP32-CAM and custom ESP32 smartwatch
# ═══════════════════════════════════════════════════════════════
def show_device_sidebar():
    st.sidebar.markdown(f"### 👤 {st.session_state.username}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in ('logged_in', 'user_id', 'username', 'token'):
            st.session_state[k] = False if k == 'logged_in' else None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 🔌 Devices")

    r = api("/api/devices/list")
    devices = r.json().get('devices', []) if (r is not None and r.status_code == 200) else []

    cams = [d for d in devices if d['device_type'] == 'esp32_cam' and d['is_active']]
    watches = [d for d in devices if d['device_type'] == 'esp32_watch' and d['is_active']]

    with st.sidebar.expander("📷 ESP32-CAM", expanded=len(cams) == 0):
        if cams:
            for d in cams:
                st.write(f"**{d['name']}** — `{d['url']}`")
                if st.button("Remove", key=f"rm_cam_{d['id']}"):
                    api(f"/api/devices/{d['id']}", "DELETE")
                    st.rerun()
        else:
            st.caption("Flash the standard CameraWebServer sketch to your ESP32-CAM, then enter its stream URL below (usually `http://<esp32-ip>:81/stream`).")
            cam_name = st.text_input("Name", value="ESP32-CAM", key="cam_name")
            cam_url = st.text_input("Stream URL", placeholder="http://192.168.1.50:81/stream", key="cam_url")
            if st.button("Connect camera", use_container_width=True):
                if cam_url:
                    rr = api("/api/devices/register", "POST",
                             json={"device_type": "esp32_cam", "name": cam_name, "url": cam_url})
                    if rr is not None and rr.status_code == 201:
                        st.success("Camera connected")
                        st.rerun()
                    else:
                        st.error("Could not register the camera")
                else:
                    st.warning("Enter the stream URL first")

    with st.sidebar.expander("⌚ ESP32 Smartwatch", expanded=len(watches) == 0):
        if watches:
            for d in watches:
                st.write(f"**{d['name']}**")
                st.caption(f"Device token (paste into the .ino sketch):")
                st.code(d['device_token'], language=None)
                st.caption(f"Last seen: {d['last_seen'] or 'never'}")
                if st.button("Remove", key=f"rm_watch_{d['id']}"):
                    api(f"/api/devices/{d['id']}", "DELETE")
                    st.rerun()
        else:
            st.caption("Register a watch to get a device token, then paste it into `esp32_smartwatch.ino` so the watch can push heart rate / SpO2 / stress readings.")
            watch_name = st.text_input("Name", value="My Smartwatch", key="watch_name")
            if st.button("Generate device token", use_container_width=True):
                rr = api("/api/devices/register", "POST",
                         json={"device_type": "esp32_watch", "name": watch_name})
                if rr is not None and rr.status_code == 201:
                    st.success(f"Device token: `{rr.json()['device_token']}`")
                    st.rerun()
                else:
                    st.error("Could not register the watch")

    return cams, watches


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    cams, watches = show_device_sidebar()

    if not HTML_PATH.exists():
        st.error(f"Could not find {HTML_PATH.name} next to streamlit_app.py")
    else:
        html = HTML_PATH.read_text(encoding="utf-8")
        cam_url = cams[0]['url'] if cams else ""
        config_script = f"""
        <script>
          window.FORMCHECK_API_URL = {st.session_state.api_url!r};
          window.FORMCHECK_DF_URL = {st.session_state.df_url!r};
          window.FORMCHECK_TOKEN = {st.session_state.token!r};
          window.FORMCHECK_ESP32_CAM_URL = {cam_url!r} || null;
        </script>
        """
        # Inject right after <head> so it runs before the dashboard's own <script> blocks.
        html = html.replace("<head>", "<head>\n" + config_script, 1)
        st.components.v1.html(html, height=1000, scrolling=True)
