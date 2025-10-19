from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from speech1 import analyze_audio_file 

# -------------------------
# Settings
# -------------------------
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# -------------------------
# Routes
# -------------------------
@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]

    # Save temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            audio_path = tmp.name
    except Exception as e:
        return jsonify({"error": f"Error saving file: {str(e)}"}), 500

    try:
        # Call the core analysis function from speech1.py
        result = analyze_audio_file(audio_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500
    finally:
        os.remove(audio_path)

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)