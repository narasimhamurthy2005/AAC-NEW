from flask import Flask, request, jsonify
from flask_cors import CORS
from fer import FER
import cv2
import numpy as np
from collections import defaultdict
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load emotion detector once (saves time)
detector = FER(mtcnn=True)

def analyze_video_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, frame_count // max_frames)

    results_per_frame = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = detector.detect_emotions(rgb_frame)

            if detected_faces:
                emotions = detected_faces[0]["emotions"]

    # Normalize to percentages
                total = sum(emotions.values())
                if total > 0:
                    emotions = {k: round((v / total) * 100, 2) for k, v in emotions.items()}

    # 🧩 Skip frames that are mostly neutral (>90%)
                if emotions.get("neutral", 0) > 90:
                    continue

    # 🎚️ Filter out weak emotions (<8%)
                emotions = {k: v for k, v in emotions.items() if v >= 8}

    # 📊 If emotions exist, record this frame
                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
                    results_per_frame.append({
                         "frame": frame_idx,
                         "time": round(frame_idx / frame_rate, 2),
                         "emotions": emotions,
                         "dominant_emotion": dominant_emotion
                })


        frame_idx += 1

    cap.release()

    # ✅ Compute overall emotion averages
    overall_emotions = defaultdict(list)
    for f in results_per_frame:
        for k, v in f["emotions"].items():
            # Reduce neutral’s influence slightly
            if k == "neutral":
                v *= 0.6
            overall_emotions[k].append(v)


    if overall_emotions:
        avg_emotions = {k: round(np.mean(v), 2) for k, v in overall_emotions.items()}
        dominant = max(avg_emotions, key=avg_emotions.get)
    else:
        avg_emotions = {}
        dominant = "None"
    # 🪄 Smooth frame-level emotions (average every 10 frames)
    smoothed = []
    window = 10
    for i in range(0, len(results_per_frame), window):
        group = results_per_frame[i:i+window]
        if not group:
            continue
        merged = defaultdict(list)
        for g in group:
            for k, v in g["emotions"].items():
                merged[k].append(v)
        smoothed.append({
            "time": group[-1]["time"],
            "emotions": {k: round(np.mean(v), 2) for k, v in merged.items()}
        })
    results_per_frame = smoothed

    return {
        "overall_emotions": [{"emotion": k, "percentage": v} for k, v in avg_emotions.items()],
        "detailed_analysis": {
            "dominant_emotion": dominant.capitalize(),
            "confidence_score": avg_emotions.get(dominant, 0)
        },
        "frames": results_per_frame
    }

@app.route('/video-emotion', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = "temp_video.mp4"

    try:
        video_file.save(video_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    try:
        print("Processing video...")
        result = analyze_video_frames(video_path)
        print(f"Processed {len(result['frames'])} frames.")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
