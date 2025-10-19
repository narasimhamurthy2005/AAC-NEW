from flask import Flask, request, jsonify, render_template
from rag_local import rag_query
# Import the sentiment analysis pipeline from Hugging Face
from transformers import pipeline

# Initialize the sentiment analysis model
# You can choose a different model if you prefer
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query')
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"Received query: {query}")

    # --- NEW LOGIC: PERFORM SENTIMENT ANALYSIS ---
    try:
        sentiment_result = sentiment_analyzer(query)[0]
        sentiment_label = sentiment_result['label'].lower()
        sentiment_score = sentiment_result['score']
        print(f"Sentiment analysis result: {sentiment_label}, score: {sentiment_score}")

        # Map the sentiment to a percentage for the frontend bars
        # This is a simplified mapping; you might need a more complex one
        emotions = {
            'joy': 0, 'sadness': 0, 'anger': 0, 'calm': 0
        }
        if sentiment_label == 'positive':
            emotions['joy'] = int(sentiment_score * 100)
        elif sentiment_label == 'negative':
            emotions['sadness'] = int(sentiment_score * 100)
        else: # Neutral
            emotions['calm'] = int(sentiment_score * 100)
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        emotions = None # Set to None if analysis fails

    # Call your RAG function
    try:
        answer = rag_query(query)
    except Exception as e:
        print(f"RAG query failed: {e}")
        answer = "🤖 An internal error occurred while processing the request."

    # Return the answer and emotion data as a JSON response
    return jsonify({
        "answer": answer,
        "emotions": emotions
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)