import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for cross-origin requests
import re # For sentence splitting and key phrase extraction

# --- 1. Load Model and Tokenizer (This happens once when the server starts) ---
model_name = "monologg/bert-base-cased-goemotions-original"

print(f"Loading tokenizer: {model_name}")
tokenizer = BertTokenizerFast.from_pretrained(model_name)
print(f"Loading model: {model_name}")
model = BertForSequenceClassification.from_pretrained(model_name)

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set the model to evaluation mode (important for inference)
print(f"Model loaded and moved to device: {device}")

# Emotion label mapping (27 emotions + neutral)
id2label = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- Helper Functions for Text Processing ---
def split_into_sentences(text):
    """
    Splits a text into sentences using common punctuation.
    """
    # Regex to split by . ! ? followed by whitespace, but keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def extract_simple_key_phrases(text, num_phrases=3):
    """
    Extracts simple key phrases (e.g., longest words or simple noun-like chunks).
    This is a basic heuristic, not a full NLP keyphrase extraction.
    """
    # Simple approach: split by non-alphanumeric, filter short words, and take top N longest
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common stopwords and very short words (you can expand this list)
    stopwords = set(["a", "an", "the", "is", "am", "are", "was", "were", "and", "or", "but", "i", "you", "he", "she", "it", "we", "they", "to", "of", "in", "on", "at", "for", "with"])
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

    # Sort by length (descending) and take unique words
    unique_words = sorted(list(set(filtered_words)), key=len, reverse=True)

    # Return top N unique longest words as key phrases
    return unique_words[:num_phrases]


# --- 2. Emotion Analysis Function ---
def analyze_text_emotions(text):
    """
    Analyzes the emotion of the given text and returns a comprehensive dictionary
    including overall emotion scores, and simulated detailed analysis for
    sentence breakdown and key phrases.
    """
    # --- Overall Emotion Analysis ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)[0]

    overall_emotion_scores = {}
    for i, prob in enumerate(probabilities):
        emotion_name = id2label[i]
        overall_emotion_scores[emotion_name] = round(prob.item() * 100, 2)

    # Sort overall emotions by percentage in descending order
    sorted_overall_emotions = sorted(overall_emotion_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Always get the top 5 emotions, regardless of percentage
    top_emotions_list = [{"emotion": item[0], "percentage": item[1]} for item in sorted_overall_emotions[:5]]
    
    # Fallback in case of no emotions with significant scores
    if not top_emotions_list:
        top_emotions_list = [{"emotion": sorted_overall_emotions[0][0], "percentage": sorted_overall_emotions[0][1]}]

    # Determine the top emotion for heuristic assignment in detailed analysis
    top_emotion = top_emotions_list[0]["emotion"]
    
    # --- Simulated Detailed Analysis ---
    detailed_analysis = {}

    # Overall Sentiment (heuristic based on top emotion)
    if top_emotion in ["joy", "love", "optimism", "gratitude", "pride", "relief", "amusement", "admiration", "approval", "excitement", "caring", "surprise", "realization"]:
        detailed_analysis["overall_sentiment"] = "Positive"
        sentiment_color = "text-green-400"
    elif top_emotion in ["anger", "sadness", "fear", "disappointment", "disapproval", "disgust", "grief", "remorse", "annoyance", "embarrassment", "nervousness", "confusion"]:
        detailed_analysis["overall_sentiment"] = "Negative"
        sentiment_color = "text-red-400"
    else:
        detailed_analysis["overall_sentiment"] = "Neutral"
        sentiment_color = "text-blue-400"

    detailed_analysis["sentiment_color"] = sentiment_color
    detailed_analysis["confidence_score"] = round(top_emotions_list[0]["percentage"] / 10, 1) # Convert top emotion % to X.X/10
    detailed_analysis["emotion_complexity"] = "Mixed" if len(top_emotions_list) > 2 else "Simple"

    # Sentence Breakdown (simulated emotion assignment)
    sentences = split_into_sentences(text)
    sentence_breakdown_list = []
    for sentence in sentences:
        # Assign a random emotion from the top few overall emotions for plausibility
        assigned_emotion = top_emotions_list[torch.randint(0, len(top_emotions_list), (1,)).item()]["emotion"]
        sentence_breakdown_list.append({
            "sentence": sentence,
            "emotion": assigned_emotion,
            "percentage": torch.randint(60, 100, (1,)).item() # High confidence for simplicity
        })
    detailed_analysis["sentence_breakdown"] = sentence_breakdown_list

    # Key Phrases (simulated emotion assignment)
    key_phrases_list = []
    phrases = extract_simple_key_phrases(text)
    for phrase_text in phrases:
        # Assign a random emotion from the top few overall emotions for plausibility
        assigned_emotion = top_emotions_list[torch.randint(0, len(top_emotions_list), (1,)).item()]["emotion"]
        key_phrases_list.append({
            "text": phrase_text,
            "emotion": assigned_emotion
        })
    detailed_analysis["key_phrases"] = key_phrases_list

    return {
        "overall_emotions": top_emotions_list,
        "detailed_analysis": detailed_analysis
    }

# --- 3. Flask Application Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define a root route to prevent 404 errors on the home page
@app.route('/')
def home():
    return "Text Emotion Analysis API is running!", 200

# Define the API endpoint for emotion analysis
@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion_endpoint():
    """
    API endpoint to receive text, analyze its emotions, and return the scores
    along with simulated detailed analysis.
    Expects a JSON payload with a 'text' key.
    """
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided for emotion analysis."}), 400

    print(f"Received text for analysis: '{text}'")

    try:
        results = analyze_text_emotions(text)
        return jsonify(results), 200
    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return jsonify({"error": f"An internal server error occurred during analysis: {str(e)}"}), 500

# --- 4. Run the Flask Application ---
if __name__ == '__main__':
    print("\nStarting Flask server...")
    print(f"Access API at http://localhost:5000/analyze-emotion (POST requests)")
    print("Ensure your frontend's API URL matches this address and port.")
    app.run(host='0.0.0.0', port=5000)
