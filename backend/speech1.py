import torch
import librosa
import whisper
from transformers import pipeline
import numpy as np
import os

# -------------------------
# Settings
# -------------------------
WHISPER_MODEL = "small"
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SAMPLE_RATE = 16000

# -------------------------
# Load models (once)
# -------------------------
print("🎙 Loading Whisper ASR model...")
asr_model = whisper.load_model(WHISPER_MODEL)

print("📝 Loading text emotion classifier...")
text_emotion_classifier = pipeline(
    "text-classification",
    model=TEXT_EMOTION_MODEL,
    top_k=None
)

# -------------------------
# Core function
# -------------------------
def analyze_audio_file(file_path):
    """
    Takes a path to an audio file (.wav, .mp3, etc.),
    returns a dictionary with transcript, top emotion, confidence, and all emotions.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    max_len = 30 * sr
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    audio_tensor = torch.from_numpy(audio).float()
    mel = whisper.log_mel_spectrogram(audio_tensor)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(asr_model, mel, options)
    text = result.text.strip()

    if not text:
        return {"text": "", "top_emotion": "", "confidence": 0.0, "emotions": []}

    scores = text_emotion_classifier(text)[0]
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    top_emotion = scores[0]["label"]
    top_score = scores[0]["score"]

    return {
        "text": text,
        "top_emotion": top_emotion,
        "confidence": top_score,
        "emotions": [{"label": s["label"], "score": s["score"]} for s in scores]
    }

# This test block is commented out to prevent the input prompt.
# if __name__ == "__main__":
#     test_file = input("Enter path to audio file for testing: ").strip()
#     result = analyze_audio_file(test_file)
#     print(result)