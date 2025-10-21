from flask import Flask, request, jsonify
from flask_cors import CORS
from fer import FER
import cv2
import numpy as np
from collections import defaultdict
import os
import traceback
import base64

app = Flask(__name__)
CORS(app)

# Load emotion detector once (saves time)
# Ensure FER is initialized with the proper face detector (mtcnn=True is a good default)
detector = FER(mtcnn=True)

def analyze_video_frames(video_path, max_frames=200):
    """Analyzes a full video file, sampling frames up to max_frames."""
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
                    frame_idx += 1
                    continue # Continue loop, skip appending

                # 🎚️ Filter out weak emotions (<8%)
                emotions = {k: v for k, v in emotions.items() if v >= 8}

                # 📊 If emotions exist, record this frame
                if emotions:
                    # dominant_emotion is not strictly needed for the smooth window but kept for completeness
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

def analyze_single_frame(frame_data):
    """Analyzes a single frame (image) for real-time analysis."""
    try:
        # Decode base64 image data
        image_data = base64.b64decode(frame_data.split(',')[1])
        # Convert to numpy array
        np_arr = np.frombuffer(image_data, np.uint8)
        # Decode image using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_emotions(rgb_frame)

        if detected_faces:
            emotions = detected_faces[0]["emotions"]

            # Normalize to percentages
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: round((v / total) * 100, 2) for k, v in emotions.items()}

            # Filter out weak emotions (<8%) and highly neutral frames
            if emotions.get("neutral", 0) > 90:
                return {} # Return empty if frame is mostly neutral

            emotions = {k: v for k, v in emotions.items() if v >= 8}

            if emotions:
                return emotions
            
        return {} # Return empty if no faces detected or no strong emotions

    except Exception as e:
        print(f"Error analyzing single frame: {e}")
        return {}


@app.route('/video-emotion', methods=['POST'])
def analyze_video():
    """Endpoint for full video file analysis."""
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

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """NEW Endpoint for real-time single frame analysis."""
    data = request.get_json()
    frame_data = data.get('frame_data')

    if not frame_data:
        return jsonify({"error": "No frame data provided"}), 400

    emotions = analyze_single_frame(frame_data)
    
    if emotions:
        return jsonify({"emotions": emotions})
    else:
        # Return an empty set of emotions if detection failed or neutral/weak
        return jsonify({"emotions": {}}) 


if __name__ == "__main__":
    app.run(debug=True, port=5000)
