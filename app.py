"""
EmoTune Tamil — Flask Backend
Face detection  : YOLOv8-face  (high precision, multi-scale, handles angles)
Emotion         : Custom Keras  emotion_model.h5
Gender / Age    : DeepFace
"""

import base64
import io
import random
import numpy as np
import os
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from deepface import DeepFace
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ─── Model Loading (once at startup) ──────────────────────────────────────────
print("[EmoTune] Loading emotion model …")
emotion_model = load_model("emotion_model.h5")
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

print("[EmoTune] Loading YOLOv8-face detector …")
yolo_face = YOLO("yolov8n-face.pt")
# Warm-up pass so first real request isn't slow
_dummy = np.zeros((64, 64, 3), dtype=np.uint8)
yolo_face(_dummy, conf=0.25, verbose=False)
print("[EmoTune] Models ready ✓")

# ─── YOLO face detection helper ────────────────────────────────────────────────
def detect_faces_yolo(frame_bgr: np.ndarray,
                      conf_threshold: float = 0.35,
                      iou_threshold:  float = 0.45) -> list[dict]:
    """
    Run YOLOv8-face on the BGR frame.
    Returns list of dicts: {x, y, w, h, det_conf}
    sorted by confidence (highest first).
    """
    results = yolo_face(
        frame_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        imgsz=640,          # YOLOv8 native resolution — best accuracy
    )

    faces = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return faces

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det_conf = float(box.conf[0])

        # Clamp to frame bounds
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))
        w  = x2 - x1
        h  = y2 - y1

        if w < 20 or h < 20:   # skip tiny spurious boxes
            continue

        faces.append({"x": x1, "y": y1, "w": w, "h": h, "det_conf": round(det_conf, 3)})

    faces.sort(key=lambda f: f["det_conf"], reverse=True)
    return faces


# ─── Emotion prediction helper ─────────────────────────────────────────────────
def predict_emotion(face_gray_roi: np.ndarray) -> tuple[str, float]:
    """
    face_gray_roi : grayscale crop, any size → resized to 48×48 internally.
    Returns (emotion_label, confidence_pct).
    """
    roi = cv2.resize(face_gray_roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 48, 48, 1)
    probs = emotion_model.predict(roi, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return EMOTION_LABELS[idx], round(float(probs[idx]) * 100, 1)


# ─── Tamil Song Database ───────────────────────────────────────────────────────
TAMIL_SONGS = {
    "male": {
        "Happy": [
            {"title": "Kavaalaa",          "movie": "Jailer (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=2Wy8xEkzUgU"},
            {"title": "Chilla Chilla",     "movie": "Thunivu (2023)",          "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=B-ZO-GI7_MM"},
            {"title": "Vaathi Coming",     "movie": "Master (2021)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=7qXnrRUo5eI"},
            {"title": "Arabic Kuthu",      "movie": "Beast (2022)",            "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=AGXtbiFmYk4"},
            {"title": "Jigidi Killaadi",   "movie": "Jailer (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=KcMKoD3fGwc"},
            {"title": "Enjoy Enjaami",     "movie": "Single (2021)",           "singer": "Dhee ft. Arivu",        "youtube": "https://www.youtube.com/watch?v=XBcMPMxcMEk"},
            {"title": "Whistle Podu",      "movie": "Leo (2023)",              "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=5gFHVWZqO-E"},
            {"title": "Mango Mango",       "movie": "Naa Saami Ranga (2024)", "singer": "Devi Sri Prasad",        "youtube": "https://www.youtube.com/watch?v=6AAAAAAA001"},
        ],
        "Sad": [
            {"title": "Nenjame",           "movie": "Custody (2023)",          "singer": "Sid Sriram",            "youtube": "https://www.youtube.com/watch?v=J8OeKmgFwB4"},
            {"title": "Kannaana Kanney",   "movie": "Viswasam (2019)",         "singer": "D. Imman",              "youtube": "https://www.youtube.com/watch?v=xRGIRv_4uRo"},
            {"title": "Inaindha Neram",    "movie": "Soorarai Pottru (2020)", "singer": "G.V. Prakash Kumar",     "youtube": "https://www.youtube.com/watch?v=ZN_4Y8MMHSE"},
            {"title": "Vaa Vaathi",        "movie": "Vaathi (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=bqoNTKf87bY"},
            {"title": "Yaar Yaar Sivakasi","movie": "Pathu Thala (2023)",      "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=1oC-7t5Dn1Q"},
            {"title": "Mazhai Kuruvi",     "movie": "PS-1 (2022)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=I3OYOLGolzw"},
        ],
        "Angry": [
            {"title": "Rolex Theme",       "movie": "Vikram (2022)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=7N0t8mh_4TM"},
            {"title": "Thugs",             "movie": "Leo (2023)",              "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=rLxfGCKPnT8"},
            {"title": "Simba Theme",       "movie": "Jailer (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=GfMFG3eDQWk"},
            {"title": "Beast Mode",        "movie": "Beast (2022)",            "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=4BqQNnZ0Ceg"},
            {"title": "Badass Ravanan",    "movie": "Leo (2023)",              "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=SVZFY17LNHM"},
        ],
        "Neutral": [
            {"title": "Naan Un Azhaginile","movie": "Ponniyin Selvan (2022)",  "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=TtFqFyFqT-U"},
            {"title": "Roja Kaadhal",      "movie": "PS-2 (2023)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=jVJ7f76V7Ks"},
            {"title": "Anbe En Anbe",      "movie": "Vendhu Thanindhathu (2022)","singer": "A.R. Rahman",         "youtube": "https://www.youtube.com/watch?v=n3SRhyVU_XQ"},
            {"title": "Megam Karukatha",   "movie": "Ponniyin Selvan (2022)",  "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=SHT5mDiGSV4"},
        ],
        "Fear": [
            {"title": "Surviva",           "movie": "Vivegam (2017)",          "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=eAFHIgmKYLw"},
            {"title": "Petta Rap",         "movie": "Petta (2019)",            "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=O_kbwlv4JAw"},
            {"title": "Rolex BGM",         "movie": "Vikram (2022)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=7N0t8mh_4TM"},
        ],
        "Surprise": [
            {"title": "Othaiyadi Pathayila","movie": "96 (2018)",              "singer": "Govind Vasantha",       "youtube": "https://www.youtube.com/watch?v=gWJFCE_DxuI"},
            {"title": "Zinda Banda",       "movie": "Jawan Tamil (2023)",      "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=CkFBg04dDaA"},
            {"title": "Kanmoodi",          "movie": "Soorarai Pottru (2020)", "singer": "G.V. Prakash Kumar",     "youtube": "https://www.youtube.com/watch?v=9S0PvS9IQZA"},
        ],
        "Disgust": [
            {"title": "Local Boys",        "movie": "Varisu (2023)",           "singer": "Thaman S",              "youtube": "https://www.youtube.com/watch?v=7YxFg8WXAQ8"},
            {"title": "Puli Urumudhu",     "movie": "Jailer (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=hXS8XMOL7j4"},
        ],
    },
    "female": {
        "Happy": [
            {"title": "Naan Pizhai",       "movie": "Jailer (2023)",           "singer": "Shreya Ghoshal",        "youtube": "https://www.youtube.com/watch?v=TBsW7gF2oWw"},
            {"title": "Oo Antava",         "movie": "Pushpa Tamil (2021)",     "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=JDJQe0FWIaE"},
            {"title": "Thalapakatti",      "movie": "Vaathi (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=3DZfkqaKnJo"},
            {"title": "Kannazhaga",        "movie": "3 (2012)",                "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=g8e-6wAUJWI"},
            {"title": "Enjoy Enjaami",     "movie": "Single (2021)",           "singer": "Dhee ft. Arivu",        "youtube": "https://www.youtube.com/watch?v=XBcMPMxcMEk"},
            {"title": "Arabic Kuthu",      "movie": "Beast (2022)",            "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=AGXtbiFmYk4"},
        ],
        "Sad": [
            {"title": "Nenjukkul Peidhidum","movie": "VTV (2010)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=PpXwFv2bN4g"},
            {"title": "Munbe Vaa",         "movie": "Sillunu Oru Kaadhal (2006)","singer": "A.R. Rahman",         "youtube": "https://www.youtube.com/watch?v=QqBwNqTDvYM"},
            {"title": "Azhagiye",          "movie": "KVRK (2022)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=X17lM2fCKbo"},
            {"title": "Mazhai Kuruvi",     "movie": "PS-1 (2022)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=I3OYOLGolzw"},
            {"title": "Nenjame",           "movie": "Custody (2023)",          "singer": "Sid Sriram",            "youtube": "https://www.youtube.com/watch?v=J8OeKmgFwB4"},
        ],
        "Angry": [
            {"title": "Kaala Kaala",       "movie": "Ponniyin Selvan (2022)",  "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=W7NJCbKI2_M"},
            {"title": "Lady Don",          "movie": "Single (2023)",           "singer": "G.V. Prakash",          "youtube": "https://www.youtube.com/watch?v=lNezaKAuZBs"},
            {"title": "Puli Urumudhu",     "movie": "Jailer (2023)",           "singer": "Anirudh Ravichander",   "youtube": "https://www.youtube.com/watch?v=hXS8XMOL7j4"},
        ],
        "Neutral": [
            {"title": "Venmathi",          "movie": "Minnale (2001)",          "singer": "Harris Jayaraj",        "youtube": "https://www.youtube.com/watch?v=oCuSaW6TQDY"},
            {"title": "Sollai Sol",        "movie": "Ponniyin Selvan (2022)",  "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=OiDAV91Pgfc"},
            {"title": "Ninaithale Inikkum","movie": "PS-2 (2023)",             "singer": "A.R. Rahman",           "youtube": "https://www.youtube.com/watch?v=bHfNHv64Mao"},
        ],
        "Fear": [
            {"title": "Uyire",             "movie": "Single (2022)",           "singer": "Sid Sriram",            "youtube": "https://www.youtube.com/watch?v=5lZ_TJhyJqA"},
            {"title": "Yaaro Ivan",        "movie": "Vikram Vedha (2017)",     "singer": "Sam C.S.",              "youtube": "https://www.youtube.com/watch?v=4cUKMSIhFaA"},
        ],
        "Surprise": [
            {"title": "Kanmoodi",          "movie": "Soorarai Pottru (2020)", "singer": "G.V. Prakash Kumar",     "youtube": "https://www.youtube.com/watch?v=9S0PvS9IQZA"},
            {"title": "Othaiyadi Pathayila","movie": "96 (2018)",              "singer": "Govind Vasantha",       "youtube": "https://www.youtube.com/watch?v=gWJFCE_DxuI"},
        ],
        "Disgust": [
            {"title": "Semma Weightu",     "movie": "Rajini Murugan (2016)",   "singer": "D. Imman",              "youtube": "https://www.youtube.com/watch?v=xVyJYpJ4xUs"},
            {"title": "Single Pasanga",    "movie": "Single (2022)",           "singer": "G.V. Prakash",          "youtube": "https://www.youtube.com/watch?v=6JFYbVpERBM"},
        ],
    }
}


def get_songs(emotion: str, gender: str) -> list:
    key = "female" if gender and gender.lower() in ["woman", "female"] else "male"
    pool = TAMIL_SONGS.get(key, {}).get(emotion, [])
    if not pool:
        pool = TAMIL_SONGS["male"]["Happy"]
    return random.sample(pool, min(3, len(pool)))


# ─── API Endpoints ─────────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64 → numpy BGR frame
        raw = data["image"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── YOLO face detection ──────────────────────────────────────────────
        yolo_conf  = float(data.get("conf", 0.35))
        faces_meta = detect_faces_yolo(frame_bgr, conf_threshold=yolo_conf)

        if not faces_meta:
            return jsonify({"faces": [], "message": "No face detected — try better lighting or move closer"})

        results_list = []
        for fm in faces_meta:
            x, y, w, h = fm["x"], fm["y"], fm["w"], fm["h"]

            # ── Emotion (custom Keras model) ─────────────────────────────────
            face_gray_roi  = frame_gray[y:y+h, x:x+w]
            emotion, conf  = predict_emotion(face_gray_roi)

            # ── Gender & Age (DeepFace) ──────────────────────────────────────
            face_color_roi = frame_bgr[y:y+h, x:x+w]
            gender, age    = "Unknown", "N/A"
            try:
                dfa    = DeepFace.analyze(face_color_roi, actions=["age", "gender"],
                                          enforce_detection=False, silent=True)
                gender = dfa[0]["dominant_gender"]
                age    = int(dfa[0]["age"])
            except Exception:
                pass

            songs = get_songs(emotion, gender)

            results_list.append({
                "bbox":       {"x": x, "y": y, "w": w, "h": h},
                "emotion":    emotion,
                "confidence": conf,
                "det_conf":   fm["det_conf"],   # YOLO detection confidence
                "gender":     gender,
                "age":        age,
                "songs":      songs,
            })

        return jsonify({"faces": results_list})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "detector": "YOLOv8-face"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
