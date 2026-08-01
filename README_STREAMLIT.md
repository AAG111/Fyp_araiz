# FormCheck — Streamlit + Login + Device Support

This adds a login screen and ESP32 device pairing **around** your existing
`fitness_gamified.html` dashboard. The dashboard itself — the camera view,
pose skeleton, challenges, shop, streak calendar, everything — is untouched.
It now just reads a few config values (API URL, auth token, camera source)
that Streamlit injects, and falls back to its original behaviour if those
aren't set.

## What changed vs. your original files

| File | Change |
|---|---|
| `fitness_gamified.html` | Small, surgical patches only: API/DeepFace URLs are now configurable instead of hardcoded; added an optional ESP32-CAM video source; added live smartwatch HR/SpO2/stress polling that feeds the existing stress display and challenge requests. No layout/CSS/feature changes. |
| `database.py` | Added `users`, `auth_tokens`, `devices`, `smartwatch_readings` tables and helper functions. Kept all original session/challenge functions, and added the `get_user_sessions_for_training` / `get_unique_active_dates` helpers your `server.py` was already calling but that didn't exist yet. |
| `server.py` | Same endpoints as before, now scoped to the logged-in user (via a bearer token) instead of a hardcoded `USER_ID = 1`. Added `/api/auth/*` and `/api/devices/*` endpoints. Dropped the dual-port (5000+5001) threading — run `deepface_server.py` separately if you want DeepFace, same as the report describes it as an optional sidecar. |
| `streamlit_app.py` | **New.** Login/register screen, then embeds the HTML dashboard via `st.components.v1.html`. Sidebar has a "Devices" panel to connect an ESP32-CAM and/or your custom ESP32 smartwatch. |
| `esp32_smartwatch.ino` | New example firmware for your custom ESP32 smartwatch (MAX30102 HR/SpO2 sensor) that posts readings to the new `/api/devices/smartwatch/data` endpoint. |

Your ML/data files (`adaptive_model_kaggle.py`, `challenge_engine.py`,
`load_fitness_tracker_dataset.py`, `preexisting_database_from_kaggle.py`,
`train_model.py`, `deepface_server.py`) are unchanged.

`app_with_auth.py`, `api_server.py`, and the `*_unified.py` files referenced
in your old `SETUP_GUIDE.md` are superseded by `streamlit_app.py` + `server.py`
above — you can delete them.

## Run it

```bash
pip install -r requirements.txt

# Terminal 1 — API backend
python server.py

# Terminal 2 — optional, only if you want facial-emotion/stress detection
python deepface_server.py

# Terminal 3 — the Streamlit app
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), register an
account, and you'll land on the same dashboard as before.

## Connecting devices

**ESP32-CAM** (visual pose tracking from a mounted camera instead of your
laptop webcam): flash the standard `CameraWebServer` Arduino example (as
described in your report, section 3.2.1), note the IP it prints over serial,
then in the Streamlit sidebar → Devices → ESP32-CAM, enter
`http://<esp32-ip>:81/stream`. Click **START CAM** in the dashboard as usual
— it now streams from the ESP32-CAM instead of asking for webcam permission.
Remove the device to go back to the laptop webcam.

**Custom ESP32 smartwatch**: in the sidebar → Devices → ESP32 Smartwatch,
click "Generate device token". Paste that token into `esp32_smartwatch.ino`
along with your Wi-Fi credentials and your `server.py` machine's IP, then
flash it. Once it's sending data, live BPM/SpO2 appear in the dashboard's
Vitals Summary panel and feed the AI difficulty predictions, same as the
Phase 4 smartwatch integration described in your report.

## Notes on hosting

- **Local / home network (recommended to start):** run all three processes
  on one machine; other devices on the same Wi-Fi can reach Streamlit at
  `http://<your-computer-ip>:8501`. Your ESP32-CAM and smartwatch need to be
  on the same network as `server.py`.
- **Streamlit Community Cloud:** Streamlit Cloud can host `streamlit_app.py`,
  but it can't run your Flask backend, TensorFlow model, or DeepFace sidecar
  next to it, and your ESP32 hardware on your home Wi-Fi can't reach a
  laptop that's now offline. If you want a fully hosted version, `server.py`
  needs to run on a small always-on host (a Raspberry Pi at home, or a cheap
  VPS/Render/Fly.io instance) with a stable address, and the ESP32 devices +
  Streamlit app both point at that address instead of `localhost`. The
  "Backend server address" box on the login screen is there for exactly this
  — set it once and it's remembered for your session.
- Everyone who logs in gets their own sessions, challenges, credits, and
  devices (`database.py`'s tables are all keyed by `user_id`), so this is
  ready for more than one person to use.
