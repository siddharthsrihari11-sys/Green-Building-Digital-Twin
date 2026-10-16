import os
import json
import time
import socket
import threading
import collections
import traceback
import logging

import cv2
import numpy as np
import requests

from flask import Flask, Response, jsonify, request, send_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONFIGURATION
# ============================================================

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

CAMERAS = [
    {
        "name": "Main corridor",
        "src": "sim" if DEMO_MODE else 0
    },
    {
        "name": "Lobby",
        "src": "sim" if DEMO_MODE else "remote"
    },
    {
        "name": "Lab wing",
        "src": "sim" if DEMO_MODE else "remote"
    }
]

TILES = 30

# Assumptions, not real meter measurements
WATT_PER_LIGHT = 10.0
CO2_KG_PER_KWH = 0.82
RS_PER_KWH = 8.0

AMBIENT = 0.08
VACANCY_S = 4.0
MIN_AREA = 3000
MOVE_V = 0.08

FADE_OUT_S = max(
    0.05,
    float(os.getenv("FADE_OUT_S", "1.5"))
)

OFFLINE_SAFE_LEVEL = min(
    1.0,
    max(0.0, float(os.getenv("OFFLINE_SAFE_LEVEL", "0.25")))
)

LIGHT_CONTROL_MODE = os.getenv(
    "LIGHT_CONTROL_MODE", "dry-run"
)

SECURITY_ENABLED = os.getenv(
    "SECURITY_ENABLED", "1"
) == "1"

SECURITY_START_HOUR = int(
    os.getenv("SECURITY_START_HOUR", "22")
)

SECURITY_END_HOUR = int(
    os.getenv("SECURITY_END_HOUR", "6")
)

SECURITY_EVENT_COOLDOWN = max(
    0.0,
    float(os.getenv("SECURITY_EVENT_COOLDOWN", "30"))
)

AI_BACKEND = os.getenv("AI_BACKEND", "rules")

security_events = collections.deque(maxlen=200)
last_security_event = {}

lock = threading.Lock()

cams = {
    i: {
        "name": camera["name"],
        "online": False,
        "position": 0.5,
        "velocity": 0.0,
        "direction": "STATIONARY",
        "occupancy": 0,
        "grid": [0.0] * TILES,
        "watts": 0.0,
        "baseline_watts": TILES * WATT_PER_LIGHT,
        "kwh": 0.0,
        "baseline_kwh": 0.0
    }
    for i, camera in enumerate(CAMERAS)
}

jpegs = {}
remote = {}

# Samples: timestamp, occupancy per zone, power per zone
history = collections.deque(maxlen=900)

insight = {
    "text": "Collecting data. First analysis in about 10 seconds.",
    "source": "rules",
    "ts": 0
}


# ============================================================
# NETWORK HELPERS
# ============================================================

def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


USE_HTTPS = os.getenv("HTTPS") == "1"
SCHEME = "https" if USE_HTTPS else "http"
LAN_URL = f"{SCHEME}://{local_ip()}:5000"


# ============================================================
# SECURITY EVENT LOGGING
# ============================================================

def is_restricted_time(now=None):
    """Check local time, including time windows crossing midnight."""
    current = time.localtime(now or time.time())
    hour = current.tm_hour

    if SECURITY_START_HOUR == SECURITY_END_HOUR:
        return True

    if SECURITY_START_HOUR > SECURITY_END_HOUR:
        return (
            hour >= SECURITY_START_HOUR
            or hour < SECURITY_END_HOUR
        )

    return (
        SECURITY_START_HOUR
        <= hour
        < SECURITY_END_HOUR
    )


def record_security_event(cam_id, occupancy, now):
    """
    Record an unverified occupancy alert during restricted hours.
    This is not a confirmed intrusion detector.
    """
    if not SECURITY_ENABLED:
        return

    if occupancy <= 0 or not is_restricted_time(now):
        return

    previous = last_security_event.get(cam_id, 0)

    if now - previous < SECURITY_EVENT_COOLDOWN:
        return

    event = {
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(now)
        ),
        "camera_id": cam_id,
        "zone": CAMERAS[cam_id]["name"],
        "type": "occupancy_during_restricted_hours",
        "occupancy": int(occupancy),
        "status": "unverified_alert"
    }

    security_events.appendleft(event)
    last_security_event[cam_id] = now

    logging.warning("Security alert: %s", event)


# ============================================================
# LIGHTING HARDWARE BOUNDARY
# ============================================================

def publish_light_levels(cam_id, levels):
    """
    Dry-run interface for future physical lighting integration.

    No physical lights are controlled here.
    A real adapter must be designed for the selected controller
    and tested independently before being enabled.
    """
    if LIGHT_CONTROL_MODE != "dry-run":
        logging.warning(
            "Refusing unsupported lighting mode: %s",
            LIGHT_CONTROL_MODE
        )
        return

    # Simulation only. No GPIO or mains control.
    return


# ============================================================
# CAMERA HELPERS
# ============================================================

def blank(message):
    frame = np.full((360, 640, 3), 235, dtype=np.uint8)

    cv2.putText(
        frame,
        message,
        (30, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (90, 90, 90),
        2
    )

    return frame


def sim_frame(cam_id, current_time):
    """Generate a synthetic person moving through a camera view."""
    frame = np.full((360, 640, 3), 190, dtype=np.uint8)

    phase = (current_time + cam_id * 5) % 20
    x = None

    if 2 <= phase < 8:
        x = 0.05 + 0.9 * (phase - 2) / 6
    elif 8 <= phase < 11:
        x = 0.95
    elif 11 <= phase < 17:
        x = 0.95 - 0.9 * (phase - 11) / 6

    if x is not None:
        cx = int(x * 640)

        cv2.ellipse(
            frame,
            (cx, 240),
            (34, 90),
            0,
            0,
            360,
            (40, 40, 45),
            -1
        )

        cv2.circle(
            frame,
            (cx, 125),
            24,
            (40, 40, 45),
            -1
        )

    cv2.putText(
        frame,
        "SIMULATED",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 120, 120),
        1
    )

    return frame


# ============================================================
# COMPUTER VISION AND LIGHTING ENGINE
# ============================================================

def worker(cam_id):
    source = CAMERAS[cam_id]["src"]

    jpegs[cam_id] = cv2.imencode(
        ".jpg", blank("STARTING CAMERA")
    )[1].tobytes()

    if isinstance(source, int):
        cap = (
            cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if os.name == "nt"
            else cv2.VideoCapture(source)
        )
    elif str(source).startswith("http"):
        cap = cv2.VideoCapture(source)
    else:
        cap = None

    logging.info(
        "Camera %s source=%r",
        cam_id,
        source
    )

    background = None
    tracks = {}
    next_track_id = 0
    last_seen = 0.0
    grid = [0.0] * TILES
    previous_time = time.time()

    while True:
        now = time.time()
        dt = max(now - previous_time, 0.001)
        previous_time = now

        live = True

        if source == "sim":
            frame = sim_frame(cam_id, now)

        elif source == "remote":
            image, timestamp = remote.get(
                cam_id, (None, 0)
            )

            live = (
                image is not None
                and now - timestamp < 5
            )

            frame = (
                cv2.resize(image, (640, 360))
                if live
                else blank("AWAITING REMOTE CAMERA")
            )

        else:
            ok, frame = (
                cap.read()
                if cap is not None and cap.isOpened()
                else (False, None)
            )

            live = bool(ok)

            if live:
                frame = cv2.resize(frame, (640, 360))

                if isinstance(source, int):
                    frame = cv2.flip(frame, 1)
            else:
                frame = blank("CAMERA OFFLINE")
                time.sleep(0.2)

        # Motion-based foreground detection.
        # This is not a trained person-recognition model.
        detections = []

        if live:
            gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (35, 35),
                0
            ).astype("float32")

            if background is None:
                background = gray.copy()

            difference = cv2.convertScaleAbs(
                cv2.absdiff(gray, background)
            )

            mask = cv2.threshold(
                difference,
                22,
                255,
                cv2.THRESH_BINARY
            )[1]

            mask = cv2.dilate(
                mask, None, iterations=4
            )

            if mask.mean() > 150:
                background = gray.copy()
            else:
                cv2.accumulateWeighted(
                    gray,
                    background,
                    0.03,
                    mask=cv2.bitwise_not(mask)
                )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in sorted(
                contours,
                key=cv2.contourArea,
                reverse=True
            )[:4]:

                if cv2.contourArea(contour) > MIN_AREA:
                    bx, by, bw, bh = cv2.boundingRect(
                        contour
                    )

                    x = (bx + bw / 2) / 640

                    detections.append(
                        (x, (bx, by, bw, bh))
                    )
        else:
            background = None

        # Approximate tracking using nearest horizontal position.
        used = set()

        for x, box in detections:
            bx, by, bw, bh = box

            best_id = None
            best_distance = 0.25

            for track_id, track in tracks.items():
                distance = abs(track["x"] - x)

                if (
                    track_id not in used
                    and distance < best_distance
                ):
                    best_id = track_id
                    best_distance = distance

            if best_id is None:
                best_id = next_track_id
                next_track_id += 1

                tracks[best_id] = {
                    "x": x,
                    "v": 0.0,
                    "seen": now
                }

            track = tracks[best_id]
            used.add(best_id)

            old_x = track["x"]
            track["x"] += (x - old_x) * 0.5

            measured_velocity = (
                track["x"] - old_x
            ) / dt

            track["v"] = (
                0.6 * track["v"]
                + 0.4 * measured_velocity
            )

            track["seen"] = now

            cv2.rectangle(
                frame,
                (bx, by),
                (bx + bw, by + bh),
                (0, 255, 204),
                2
            )

            cv2.putText(
                frame,
                f"ID {best_id}",
                (bx, max(by - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 204),
                1
            )

        # Remove tracks that have not been observed recently.
        tracks = {
            track_id: track
            for track_id, track in tracks.items()
            if now - track["seen"] < 1.0
        }

        if tracks:
            last_seen = now

        # If the camera fails, occupancy becomes unknown.
        # Do not interpret a failed camera as an empty room.
        if not live:
            target = [OFFLINE_SAFE_LEVEL] * TILES
        elif now - last_seen < VACANCY_S:
            target = [AMBIENT] * TILES
        else:
            target = [0.0] * TILES

        # Movement-aware light distribution.
        # Each track contributes to the target intensity.
        for track in tracks.values():
            speed = abs(track["v"])
            moving = speed > MOVE_V

            sign = 1 if track["v"] > 0 else -1
            lead = min(12, 3 + speed * 10)

            position = int(
                min(TILES - 1, track["x"] * TILES)
            )

            for i in range(TILES):
                distance = i - position

                if moving:
                    ahead = distance * sign

                    if ahead >= 0:
                        if ahead <= lead:
                            level = 1.0
                        else:
                            level = max(
                                0.0,
                                1 - (ahead - lead) * 0.25
                            )
                    else:
                        level = 0.6 ** (-ahead)
                else:
                    level = max(
                        0.0,
                        1 - abs(distance) * 0.28
                    )

                target[i] = max(target[i], level)

        # Fast brightening and bounded fade-out.
        fade_alpha = min(
            1.0,
            dt / FADE_OUT_S
        )

        grid = [
            min(1.0, max(0.0, target_level))
            if target_level >= current_level
            else max(
                target_level,
                current_level * (1 - fade_alpha)
            )
            for current_level, target_level
            in zip(grid, target)
        ]

        # Ensure vacancy eventually produces a true zero.
        if (
            live
            and not tracks
            and now - last_seen >= VACANCY_S + FADE_OUT_S
        ):
            grid = [0.0] * TILES

        primary = max(
            tracks.values(),
            key=lambda track: abs(track["v"]),
            default=None
        )

        watts = sum(grid) * WATT_PER_LIGHT

        with lock:
            camera = cams[cam_id]

            camera.update(
                online=live,
                occupancy=len(tracks),
                grid=[round(level, 2) for level in grid],
                watts=round(watts, 1),
                position=(
                    round(primary["x"], 3)
                    if primary else 0.5
                ),
                velocity=(
                    round(abs(primary["v"]), 2)
                    if primary else 0.0
                ),
                direction=(
                    "RIGHT" if primary["v"] > 0
                    else "LEFT"
                ) if (
                    primary
                    and abs(primary["v"]) > MOVE_V
                ) else "STATIONARY"
            )

            # Only accumulate energy estimates when the camera
            # input is live. Offline is not proof of vacancy.
            if live:
                camera["kwh"] += (
                    watts * dt / 3.6e6
                )

                camera["baseline_kwh"] += (
                    TILES * WATT_PER_LIGHT * dt / 3.6e6
                )

        record_security_event(
            cam_id,
            len(tracks) if live else 0,
            now
        )

        publish_light_levels(cam_id, grid)

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if success:
            jpegs[cam_id] = buffer.tobytes()

        time.sleep(
            max(0, 0.04 - (time.time() - now))
        )


def safe_worker(cam_id):
    """Restart a camera-processing thread after an exception."""
    while True:
        try:
            worker(cam_id)
        except Exception:
            traceback.print_exc()

            jpegs[cam_id] = cv2.imencode(
                ".jpg",
                blank("CAMERA ERROR - CHECK TERMINAL")
            )[1].tobytes()

            time.sleep(2)


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

def summary():
    rows = list(history)[-300:]

    if len(rows) < 5:
        return None

    output = []

    with lock:
        camera_snapshot = {
            i: dict(camera)
            for i, camera in cams.items()
        }

    for i, camera in camera_snapshot.items():
        occupancy = [row[1][i] for row in rows]
        power = [row[2][i] for row in rows]

        output.append({
            "zone": camera["name"],
            "avg_people": round(
                sum(occupancy) / len(occupancy), 2
            ),
            "occupied_pct": round(
                100 * sum(p > 0 for p in occupancy)
                / len(occupancy)
            ),
            "avg_watts": round(
                sum(power) / len(power), 1
            ),
            "always_on_watts": TILES * WATT_PER_LIGHT
        })

    return output


def rule_insight(data):
    worst = max(data, key=lambda z: z["avg_watts"])
    idle = min(data, key=lambda z: z["occupied_pct"])
    busy = max(data, key=lambda z: z["avg_people"])

    usage = (
        100 * worst["avg_watts"]
        / worst["always_on_watts"]
    )

    if busy["avg_people"] == 0:
        return "\n".join([
            "Energy: No movement detected. Verify camera status before trusting the estimated saving.",
            "Sustainability: Consider daylight and ventilation scheduling while preserving indoor comfort.",
            "Resources: Review lighting and other idle electrical loads in unoccupied zones."
        ])

    return "\n".join([
        f"Energy: {worst['zone']} uses the most lighting power, at {usage:.0f}% of its assumed always-on baseline.",
        f"Sustainability: {busy['zone']} is busiest. Prioritize daylight and ventilation scheduling while maintaining comfort.",
        f"Resources: {idle['zone']} has the lowest recorded occupancy. Review its standby lighting and vacancy timeout."
    ])


def call_llm(prompt):
    if AI_BACKEND == "gemini":
        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={
                "key": os.environ["GEMINI_API_KEY"]
            },
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            timeout=20
        )

        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": os.getenv(
                "OLLAMA_MODEL", "llama3.2"
            ),
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()
    return response.json()["response"]


def run_ai():
    global insight

    data = summary()

    if not data:
        return

    recommendation = None
    source = "rules"

    if AI_BACKEND in ("gemini", "ollama"):
        prompt = (
            "You are an analytics engine for a Green Building Digital Twin. "
            "Analyze this zone-level occupancy and lighting data: "
            + json.dumps(data)
            + ". Produce exactly three concise recommendations, one per line, "
            "starting with Energy:, Sustainability:, and Resources:. "
            "Do not claim that estimated energy figures are measured."
        )

        try:
            recommendation = call_llm(prompt).strip()
            source = AI_BACKEND
        except Exception:
            logging.exception(
                "AI backend failed; using rule-based recommendations"
            )

    with lock:
        insight.update(
            text=recommendation or rule_insight(data),
            source=source,
            ts=time.time()
        )


def housekeeping():
    counter = 0

    while True:
        time.sleep(1)
        counter += 1

        with lock:
            history.append((
                time.time(),
                [camera["occupancy"] for camera in cams.values()],
                [camera["watts"] for camera in cams.values()]
            ))

        if counter == 10 or counter % 30 == 0:
            run_ai()


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return send_file(
        os.path.join(BASE, "index.html")
    )


@app.route("/state")
def state():
    with lock:
        camera_data = {
            str(i): dict(camera)
            for i, camera in cams.items()
        }

        watts = sum(
            camera["watts"] for camera in cams.values()
        )

        baseline_watts = sum(
            camera["baseline_watts"]
            for camera in cams.values()
        )

        energy = sum(
            camera["kwh"] for camera in cams.values()
        )

        baseline_energy = sum(
            camera["baseline_kwh"]
            for camera in cams.values()
        )

        occupancy = sum(
            camera["occupancy"]
            for camera in cams.values()
        )

    saved_kwh = max(0.0, baseline_energy - energy)

    building = {
        "occupancy": occupancy,
        "watts": round(watts),
        "baseline_watts": round(baseline_watts),
        "saved_pct": round(
            100 * (1 - watts / baseline_watts), 1
        ) if baseline_watts else 0,
        "wh_saved": round(saved_kwh * 1000, 2),
        "co2_g": round(
            saved_kwh * CO2_KG_PER_KWH * 1000, 2
        ),
        "rs_month": round(
            (baseline_watts - watts)
            / 1000 * 24 * 30 * RS_PER_KWH
        ),
        "note": (
            f"Estimates use {WATT_PER_LIGHT:.0f} W per light, "
            f"{CO2_KG_PER_KWH} kg CO2/kWh and "
            f"Rs {RS_PER_KWH:.0f}/kWh."
        )
    }

    with lock:
        current_insight = dict(insight)

    return jsonify(
        cams=camera_data,
        insight=current_insight,
        lan_url=LAN_URL,
        building=building
    )


def mjpeg(cam_id):
    while True:
        frame = jpegs.get(cam_id)

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.04)


@app.route("/video_feed/<int:i>")
def video_feed(i):
    if i not in cams:
        return "No such camera", 404

    return Response(
        mjpeg(i),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/remote_cam/<int:i>", methods=["POST"])
def remote_cam(i):
    if i not in cams:
        return jsonify(status="unknown camera"), 404

    image = cv2.imdecode(
        np.frombuffer(request.data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify(status="bad frame"), 400

    remote[i] = (image, time.time())

    return jsonify(status="ok")


CAM_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera _ID_</title>
<style>
body { margin:0; background:#09090b; color:#f3f3f1;
       font:18px system-ui; text-align:center; padding:20px; }
video { width:100%; max-width:560px; border-radius:8px; background:#000; }
button { font:600 18px system-ui; padding:14px 28px;
         border-radius:6px; border:0; margin:16px 0; cursor:pointer; }
#status { color:#aaa; }
</style>
</head>
<body>
<h2>Camera _ID_ - _NAME_</h2>
<video id="video" autoplay playsinline muted></video><br>
<button id="start">Start sending camera</button>
<div id="status">Tap the button and allow camera access.</div>
<script>
const id = "_ID_";
const video = document.getElementById("video");
const status = document.getElementById("status");
const canvas = document.createElement("canvas");
canvas.width = 640;
canvas.height = 360;

document.getElementById("start").onclick = async () => {
    try {
        video.srcObject = await navigator.mediaDevices.getUserMedia({
            video: {width:640, height:360},
            audio: false
        });
    } catch (error) {
        status.textContent = "Camera blocked: " + error.message;
        return;
    }

    let busy = false;
    let count = 0;

    setInterval(() => {
        if (busy || video.videoWidth === 0) return;

        busy = true;

        canvas.getContext("2d").drawImage(
            video, 0, 0, 640, 360
        );

        canvas.toBlob(async blob => {
            try {
                const response = await fetch(
                    "/api/remote_cam/" + id,
                    {method:"POST", body:blob}
                );

                if (!response.ok) {
                    throw new Error("Server rejected frame");
                }

                count++;
                status.textContent = "Sending frames: " + count;
            } catch (error) {
                status.textContent = "Cannot reach server";
            } finally {
                busy = false;
            }
        }, "image/jpeg", 0.6);
    }, 125);
};
</script>
</body>
</html>
"""


@app.route("/cam/<int:i>")
def cam_page(i):
    if i not in cams:
        return "No such camera", 404

    return CAM_PAGE.replace(
        "_ID_", str(i)
    ).replace(
        "_NAME_", cams[i]["name"]
    )


@app.route("/api/security/events")
def security_event_feed():
    with lock:
        events = list(security_events)

    return jsonify(
        enabled=SECURITY_ENABLED,
        restricted_hours={
            "start": SECURITY_START_HOUR,
            "end": SECURITY_END_HOUR
        },
        events=events
    )


@app.route("/api/ai/run", methods=["POST"])
def ai_run():
    threading.Thread(
        target=run_ai,
        daemon=True
    ).start()

    return jsonify(status="started")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    for camera_id in cams:
        threading.Thread(
            target=safe_worker,
            args=(camera_id,),
            daemon=True
        ).start()

    threading.Thread(
        target=housekeeping,
        daemon=True
    ).start()

    print()
    print("Dashboard:", f"{SCHEME}://localhost:5000")
    print("LAN URL:", LAN_URL)
    print("Demo mode:", DEMO_MODE)
    print("Lighting control:", LIGHT_CONTROL_MODE, "(dry-run)")
    print(
        "Security alerts:",
        SECURITY_ENABLED,
        f"({SECURITY_START_HOUR:02d}:00–{SECURITY_END_HOUR:02d}:00)"
    )

    for i, camera in enumerate(CAMERAS):
        if camera["src"] == "remote":
            print(
                f"Camera {i} sender page: {LAN_URL}/cam/{i}"
            )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        ssl_context="adhoc" if USE_HTTPS else None
    )